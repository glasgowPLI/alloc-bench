#!/usr/bin/python3 

import sys 
import json 
import pathlib
import argparse 

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

out_formats = ["pdf", "png", "svg"]


import numpy as np


instr_class_map = { 
  "branches"  : [
      "PCrel addr", 
      "Cond Branch (imm)", 
      "Exception Gen", 
      "Hints", 
      "Barriers", 
      "Branch (reg)", 
      "Branch (imm)", 
      "Cmp & Branch", 
      "Tst & Branch", 
    ], 
  "data proc" : [
      "Add/Sub (imm)", 
      "Logical (imm)", 
      "Move Wide (imm)", 
      "Bitfield", 
      "Extract", 
      "Data Proc Reg"
    ], 
  "ld/st" : [
      "AdvSimd ldstmult", 
      "AdvSimd ldstmult++", 
      "AdvSimd ldst", 
      "AdvSimd ldst++", 
      "ldst excl", 
      "Load Reg (lit)", 
      "ldst pair", 
      "ldst reg (imm)", 
      "Loads & Stores", 
    ], 
  "Scalar FP" : [
      "Scalar FP"
    ], 
  "morello arith" : [
      "morello add/sub cap"
    ], 
  "morello ld/st" : [
      "morello ld/st misc1", 
      "morello ld/st misc2",
      "morello ld/st misc3",
      "morello ld/st uovab",
      "morello ld/st misc4",
      "morello ld/st uo"
    ], 
  "morello misc" : [
      "morello misc"
    ], 
  "morello regs" : [
      "morello sysreg", 
      "morello add extreg", 
      "PSTATE", 
      "System Insn", 
      "System Reg",
    ]
} 



instr_colours = { 
    'data proc'     : 'dimgray', 
    'Scalar FP'     : 'salmon', 
    'branches'      : 'turquoise', 
    'ld/st'         : 'sandybrown', 
    'morello arith' : 'lightseagreen', 
    'morello ld/st' : 'cadetblue', 
    'morello misc'  : 'skyblue', 
    'morello regs'  : 'dodgerblue'
    #'spare unused 1' : 'silver', 
    #'spare unused 2' : 'whitesmoke', 
}

dev_modes = ["hybrid", "hybrid_nc", "purecap", "benchmarkabi"]


# Open data log file, extract records and remove headers 
def extract_records(filename): 
  # Read data from file
  with open(f"{filename}") as fd: 
    _lines = fd.readlines()
    if _lines[0] == "Instruction Classes:\n":
      t = _lines.pop(0)
    return _lines
  return []


# Convenience function to split and parse log file to obtain data
def get_instr_class(columns): 
  _iclass = "".join(columns[0][len('Class:'):]).lstrip()
  if _iclass in ["NOP", "UDEF"]: 
    return (_iclass, None) 

  _user_instr = "".join(columns[1].split(sep=',')[0][len('hits: user -)'):]).lstrip()
  return (_iclass, int(_user_instr))

 
def reverse_map( orig_map ):
  rmap = {}
  for key, val in orig_map.items():
    assert len(val) > 0, f"Zero list"
    for _nkey in val:
      assert _nkey not in rmap, f"Key {_nkey} already in reverse map"
      rmap[_nkey] = key
  return rmap

# Convert log file record format to JSON     
def convert_to_json(records): 
  # remove white space and convert instructions to 
  instr_rec = {}
  for _rec in records: 
    fmt_rec = get_instr_class([ _r.strip() for _r in _rec.split(sep='\t')])
    if fmt_rec[1] is None: 
      continue
    assert not fmt_rec[0] in instr_rec, f"isntruction class: {fmt_rec[0]} already found when processing {config}-{bm}"
    instr_rec[fmt_rec[0]] = fmt_rec[1]
  return instr_rec
 

# Consolidate and make instruction classes coarser for graphing 
def consolidate_instr_classes( base_data, instr_mix ): 
  r_instr_class_map = reverse_map(instr_class_map) 
 
  consolidated_base_instr = {} 
  for iclass in base_data.keys(): 
    consolidated_iclass = r_instr_class_map[iclass] 
    if consolidated_iclass in consolidated_base_instr: 
      consolidated_base_instr[consolidated_iclass] += base_data[iclass] 
    else: 
      consolidated_base_instr[consolidated_iclass] = base_data[iclass] 


  consolidated_instr_mix = { _conf: { _bm : { } for _bm in instr_mix[_conf].keys() 
                                    } for _conf in instr_mix.keys()
                           } 

  for conf in instr_mix.keys(): 
    for bm in instr_mix[conf].keys():
      # consolidate instruction classes into coarser groups 
      for iclass in instr_mix[conf][bm].keys(): 
        consolidated_iclass = r_instr_class_map[iclass] 
        if consolidated_iclass in consolidated_instr_mix[conf][bm]: 
          consolidated_instr_mix[conf][bm][consolidated_iclass] += instr_mix[conf][bm][iclass] 
        else: 
          consolidated_instr_mix[conf][bm][consolidated_iclass] = instr_mix[conf][bm][iclass] 

      # deduct kernel boot-up and shutdown sequence instruction counts
      for consolidated_class in consolidated_instr_mix[conf][bm].keys(): 
        consolidated_instr_mix[conf][bm][consolidated_class] -= consolidated_base_instr[consolidated_class] 

  return consolidated_instr_mix


# Convert data from format supplied in QEMU memhowvec log format to json
def convert_data(repo): 
  assert not repo.outfile.exists() , f"output file {repo.outfile} exists already. Might over-write. Remove and try again!"

  # First obtain baseline data 
  assert repo.baseline.exists() , f"baseline file {repo.baseline} does not exist!"
  baseline_records = extract_records(repo.baseline) 
  baseline_data = convert_to_json(baseline_records) # consolidate this data later

  data = { }  # 
  for i, _infile in enumerate(repo.infile_list): 
    # Extract configuration and benchmark name 
    _infile_comp = _infile.stem.split(sep='_')
    if _infile_comp[0] == 'hybrid' and _infile_comp[1] == 'nc': 
      _config, _bm = (f'{"_".join(_infile_comp[0:2])}', f'{"_".join(_infile_comp[2:])}')
    else: 
      _config, _bm = (f'{_infile_comp[0]}', f'{"_".join(_infile_comp[1:])}')

    # Create json fields in final data structure
    if _config not in data: 
      data[_config] = {} 

    assert _bm not in data[_config], f"{_config}-{_bm} already found within records"

    records = extract_records(_infile) 
    data[_config][_bm] = convert_to_json(records) # consolidate this data later
  return baseline_data, data 



# Generate list of locations for start of bars and tick-locations
def gen_bunched_bar_loc(obj_json, bw=0.5, sw=0.5, offset=0.5, strip_zero=False):
  tick_pos, bar_pos = ([], [ [] for _bm in obj_json.keys()])
  bin_start = offset
  num_bins = len(obj_json.keys())
  tick_labels = list(obj_json["hybrid" if "hybrid" in obj_json else "purecap"].keys())
  #print(f" benchmarks for generation - {tick_labels}") 
  #print(f" num_bins =  {num_bins}") 
  #print(f" offset =  {offset}, bw = {bw}, sw= {sw}") 
  #print("==========+")

  bin_start = offset
  for idx_i in range(len(tick_labels)):
    data_end = bin_start + (num_bins*bw)
    benchmark_end = bin_start + (num_bins*bw) + sw
    for _bin in range(num_bins): 
      bar_pos[_bin].append(bin_start + (_bin * bw) + bw/2)
    tick_pos.append((data_end + bin_start)/2) 
    bin_start = benchmark_end 

  return (tick_labels, tick_pos, bar_pos)


def normalise_to_hybrid(data, hybrid_total, mode, instr_type, benchmarks): 
  return [ float(data[mode][bm][instr_type]) * 100 / hybrid_total[bm] for bm in benchmarks ] 

  

# Plot data to pdf
def plot_data(repo, data): 
  _fig, _subplot = plt.subplots(nrows=1, ncols=1, sharex=False)
  _fig.tight_layout()


  tick_lbl, tick_x, bar_x = gen_bunched_bar_loc(data, bw=0.5, sw=0.5, offset=0.5, strip_zero=False)
  #print(f"labels = {tick_lbl}\npos_tick = {tick_x}\npos_bar = {bar_x}")

  benchmarks = list(data['purecap'].keys())
  instr_types = list(instr_class_map.keys()) 

  if repo.relative: 
    hybrid_total = { _bm : float(sum(data["hybrid"][_bm].values())) 
                       for _bm in benchmarks }

  for mode_idx, mode in enumerate(dev_modes):
    bottom = np.array([0] * len(benchmarks), dtype='f') 

    for instr_idx, _instr in enumerate(instr_types):
      perf_count = normalise_to_hybrid(data, hybrid_total, mode, _instr, benchmarks) \
                       if repo.relative else [ float(data[mode][_bm][_instr]) for _bm in benchmarks ] 
      _subplot.bar( bar_x[mode_idx], perf_count,
                    color = instr_colours[_instr], width=0.5, edgecolor='black', 
                    label = _instr if mode == 'purecap' else None
                    , bottom = bottom, hatch = '...' if _instr.startswith('morello') else '' )
      bottom += perf_count  # Stack the instruction metrics on top of the other


  # Generate final parameters for each subplot
  _subplot.grid = True 
  _subplot.set_xticks(tick_x)
  _subplot.set_xticklabels( tick_lbl , rotation=15.0, fontsize="medium")
  _subplot.legend(loc='upper left', ncol=4, fontsize='x-large')
  if repo.relative:
    _subplot.set_ylabel("Normalised (vs hybrid)", fontsize="medium")
  else:
    ax.set_ylabel('num instructions')




def render(filename, adjust=None):
  if adjust != None :
    _left, _right, _bottom, _top, _hspace = adjust
    plt.subplots_adjust(left=_left,right=_right,bottom=_bottom,top=_top,hspace=_hspace)
  else:
    plt.tight_layout()

  _suffix = filename.suffix.lstrip(".")
  assert _suffix in out_formats, f"invalid output file ({self.out_file}) format {_suffix}. " \
                                                                          f"acceptable formats -> {out_formats}"

  plt.draw()
  plt.pause(1)
  input("<Press Enter to continue>")
  print(f"saving to {filename}")
  plt.savefig(f"{filename}",format=_suffix)
  plt.close()


## Command line option processing
class CommandLine: 
  def __init__(self, name=None, desc=None, epilogue=None):
    self.parser = (name, desc, epilogue) 
    self.__gen_options()

    self.args = self.parser.parse_args() 
    self.workdir =  self.args.workdir
    self.infile_list = self.args.input
    self.outfile = self.args.output
    self.baseline = self.args.baseline
    self.relative = self.args.relative


  @property
  def parser(self):
    return self._parser

  @parser.setter
  def parser(self, prog):
    self._parser = argparse.ArgumentParser(
                              prog = sys.argv[0] if prog[0] is None else prog[0] ,
                              description = prog[1] if prog[1] is not  None else
                                                f"This utility converts the output files from multiple runs "
                                                f"of the memhowvec QEMU plugin for various CHERI configurations "
                                                f"and consolidates all such data into a single json file", 
                              epilog = prog[2] if prog[2] is not None else
                                         f"Input data is assumed to be one set of instruction classifications for a configutation"
                                         f"using cheribuild (https://github.com/CTSRD-CHERI/cheribuild)")
  @property 
  def workdir(self): 
    return self._workdir

  @workdir.setter 
  def workdir(self, path): 
    self._workdir = pathlib.Path(path).resolve()
    if not self._workdir.exists(): 
      print(f"{self._workdir} does not exist. Creating....")
      self._workdir.mkdir(mode=0o750, parents=True, exist_ok=True)
      assert False, f"Empty working directory {self._workdir} created. Input files expected here" 

  @property
  def infile_list(self): 
    return self._infiles

  @infile_list.setter
  def infile_list(self, filenames): 
    self._infiles = [ self.workdir / _f for _f in filenames ]
    for _file in self._infiles:
      assert _file.exists(), f"Error : input file {_file} does not exist" 


  @property
  def outfile(self): 
    return self._outfile

  @outfile.setter
  def outfile(self, filename): 
    self._outfile = self.workdir / filename 
    assert not self._outfile.exists(), f"Error: potential overwrite of output file" 

  @property
  def baseline(self): 
    return self._baseline 

  @baseline.setter 
  def baseline(self, filename): 
    self._baseline = self.workdir / filename 

  def __gen_options(self): 
    self.parser.add_argument('-i', '--input', nargs='*', help=f"input files from qemu runs") 
    self.parser.add_argument('-w', '--workdir', default=f"{pathlib.Path.cwd().resolve()}", 
                             help=f"directory outside source-dir to build and install benchmarks")
    self.parser.add_argument('-o', '--output', default = "out.json", 
                             help=f"output-data file name. This will be put in the working directory provided by the -w option")
    self.parser.add_argument('-b', '--baseline', default = "baseline.log", 
                             help=f"baseline log file. Should be QEMU log file with only boot sequence captured") 
    self.parser.add_argument('-v', '--verbose', action='store_true',
                             default=False, help=f"print parsed cmdline options")
    self.parser.add_argument('-r', '--relative', action='store_true',
                             default=False, help=f"Normalise all values to hybrid")
    self.parser.add_argument('-a', '--action', nargs='*',
                             choices = ['convert', 'plot'],
                             default = ['convert'],
                             help=f"actions to execute. These may be combined together to perform a sequence of events")




def verbose(repo):
  print(f"--input                   : {repo.infile_list}")
  print(f"--output                  : {repo.outfile}")
  print(f"--relative                : {repo.relative}")
  print(f"--baseline                : {repo.baseline}")
  print(f"--workdir                 : {repo.workdir}")
  print(f"--action                  : {repo.args.action}")


if __name__ == '__main__':
  repo = CommandLine()
  if repo.args.verbose: 
    verbose(repo) 

  if 'convert' in repo.args.action: 
    base_data, bm_data = convert_data(repo)
    #_jsonp = json.dumps(_jsonv, indent=2)
    #print(f"{_jsonp}")
    instr_mix  = consolidate_instr_classes(base_data, bm_data) 
    with open(f"{repo.outfile}", "w") as fd: 
      json.dump(instr_mix, fd, indent=2)


  if 'plot' in repo.args.action: 
    json_file =  repo.outfile if 'convert' in repo.args.action else repo.infile_list[0]
    outfile = repo.outfile.parent / f"{repo.outfile.stem}.pdf" if 'convert' in repo.args.action else repo.outfile
    with open(json_file) as fd: 
      json_data = json.load(fd) 
      plot_data(repo, json_data) 
      render(outfile)

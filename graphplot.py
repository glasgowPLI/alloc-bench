#!/usr/bin/python3
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import numpy as np
import json
import pathlib

import argparse
import sys
import statistics as st
import math

out_formats = ["pdf", "png", "svg"]

# The z-values were copied from the multi-time tool found at 
# Laurie Tratt's [multi-time tool](https://github.com/ltratt/multitime/blob/master/zvals.h)
zval = np.array([ 0.01253347, 0.02506891, 0.03760829, 0.05015358, 0.06270678,
                  0.07526986, 0.08784484, 0.10043372, 0.11303854, 0.12566135,
                  0.13830421, 0.15096922, 0.16365849, 0.17637416, 0.18911843,
                  0.20189348, 0.21470157, 0.22754498, 0.24042603, 0.2533471,
                  0.26631061, 0.27931903, 0.2923749, 0.30548079, 0.31863936,
                  0.33185335, 0.34512553, 0.35845879, 0.37185609, 0.38532047,
                  0.39885507, 0.41246313, 0.42614801, 0.43991317, 0.45376219,
                  0.4676988, 0.48172685, 0.49585035, 0.51007346, 0.52440051,
                  0.53883603, 0.55338472, 0.5680515, 0.58284151, 0.59776013,
                  0.61281299, 0.62800601, 0.64334541, 0.65883769, 0.67448975,
                  0.69030882, 0.70630256, 0.72247905, 0.73884685, 0.75541503,
                  0.77219321, 0.78919165, 0.80642125, 0.82389363, 0.84162123,
                  0.85961736, 0.8778963, 0.89647336, 0.91536509, 0.93458929,
                  0.95416525, 0.97411388, 0.99445788, 1.01522203, 1.03643339,
                  1.05812162, 1.08031934, 1.10306256, 1.12639113, 1.15034938,
                  1.17498679, 1.20035886, 1.22652812, 1.25356544, 1.28155157,
                  1.31057911, 1.34075503, 1.37220381, 1.40507156, 1.43953147,
                  1.47579103, 1.51410189, 1.55477359, 1.59819314, 1.64485363,
                  1.69539771, 1.75068607, 1.81191067, 1.88079361, 1.95996398,
                  2.05374891, 2.17009038, 2.32634787, 2.5758293,] )

y_labels = { "L1I_CACHE": ("L1-I-cache", "accesses"),
             "L1D_CACHE": ("L1-D-cache", "accesses"),
             "L2D_CACHE": ("L2-D-cache", "accesses"),
             "L1I_CACHE_REFILL": ("L1-I-cache", "misses"), 
             "L1D_CACHE_REFILL": ("L1-D-cache", "misses"), 
             "L2D_CACHE_REFILL": ("L2-D-cache", "misses"), 
             "LL_CACHE_RD": ("LL-cache Read", "accesses"),
             "LL_CACHE_MISS_RD": ("LL-cache Read Misses", "misses"),
             "MEM_ACCESS": ("Memory access", "accesses"),
             "BUS_ACCESS" : ("Bus-Access", "events"),
             "BUS_ACCESS_RD_CTAG" : ("Bus-Access-Ctag", "events"), 
             "CPU_CYCLES" : ("CPU-cycles", "cycles"), 
             "gc-cycles": ("GC cycles", "cycles"),
             "INST_RETIRED": ("Instructions retired", "instrs"),
             "gc-time": ("GC time", "milli-sec"), 
             "total-time": ("Total Time", "milli-sec"),
             "gc-load": ("GC load", "GC runtime ratio"),
             "rss-kb": ("RSS", "KB")
}

color_grid = { "total-time" : "cadetblue", 
               "rss-kb" : "moccasin", 
               "INST_RETIRED": "tab:purple",
               "L1D_CACHE" : "lightgreen",
               "L2D_CACHE" : "tomato",
               "L1I_CACHE" : "sandybrown",
               "L1D_CACHE_REFILL" : "lightgreen",
               "L2D_CACHE_REFILL" : "tomato",
               "L1I_CACHE_REFILL" : "sandybrown",
               "LL_CACHE_RD": "cornflowerblue",
               "LL_CACHE_MISS_RD": "turquoise",
               "MEM_ACCESS": "thistle",
               } 


march_y_labels = { "hybrid": ("Hybrid", "accesses"),
                   "hybrid_nc": ("Hybrid No-Coalescing", "accesses"),
                   "purecap": ("Purecap", "accesses"),
                   "benchmarkabi": ("Benchmark ABI", "accesses")
                   }


m_arch_colors = { "hybrid_nc" : "cadetblue", 
                 "purecap" : "tomato",
                 "benchmarkabi" : "lightgreen",
                 "hybrid" : "sandybrown",
                } 

dev_modes = ["hybrid_nc", "purecap", "benchmarkabi"] 
derived_events = ["gc-load"]

def gmean(arr):
  return np.exp(np.log(arr).mean()) if np.count_nonzero(arr) == arr.size else 0.0


def norm_conf_interval(data, mode, bm, evt, confidence=98): 
  assert mode != "hybrid", f"Normalisation of hybrid vs hybrid invalid"
  hybrid_mean = st.mean(data["hybrid"][bm][f"raw-{evt}"]) 
  hybrid_std_dev = st.stdev(data["hybrid"][bm][f"raw-{evt}"]) 
  hybrid_len = len(data["hybrid"][bm][f"raw-{evt}"]) 


  dev_mode_mean = st.mean(data[mode][bm][f"raw-{evt}"]) 
  dev_mode_std_dev = st.stdev(data[mode][bm][f"raw-{evt}"]) 
  dev_mode_len = len(data[mode][bm][f"raw-{evt}"]) 
  assert hybrid_len == dev_mode_len, f"len(hybrid raw-{evt}) [{hybrid_len}] != len({mode} raw-{evt}) [{dev_mode_len}]"
  norm_std_dev = math.sqrt(pow(hybrid_std_dev/hybrid_mean, 2) + pow(dev_mode_std_dev/dev_mode_mean, 2)) * (dev_mode_mean/hybrid_mean)
  return zval[confidence] * norm_std_dev / math.sqrt(hybrid_len) 

# Generate a list of locations for the start of bars and tick locations 
# on the bar-chart
def gen_bunched_bar_loc(obj_json, bw=0.5, sw=0.5, offset=0.5, strip_zero=False):
  tick_pos, bar_pos = ([], [ [] for _bm in obj_json.keys()])
  num_bins = len(obj_json.keys())  # hybrid-nc + purecap + benchmark-abi
  tick_labels = list(obj_json["hybrid" if "hybrid" in obj_json else "purecap"].keys())
  tick_labels += ["geo-mean"]

  # Calculate bar and tick positions
  bin_start = offset
  vline_pos = offset + ((len(tick_labels) - 1) * ((num_bins * bw) + sw)) - (sw/2)
  for idx_i in range(len(tick_labels)):
    data_end = bin_start + ((num_bins)*bw)
    benchmark_end = bin_start + ((num_bins)*bw) + sw
    for _bin in range(num_bins):
      bar_pos[_bin].append(bin_start + (_bin * bw) + bw/2)
    tick_pos.append((data_end + bin_start)/2)
    bin_start = benchmark_end 
  return (tick_labels, tick_pos, bar_pos, vline_pos)

def gen_barchart(data, event, conf_interval=95):
  temp = dict(data)
  hybrid_data = temp.pop("hybrid") 
  plot_data = temp

  tick_lbl, tick_x, bar_x, vline_barpos = gen_bunched_bar_loc(plot_data, bw=0.5, sw=0.5, offset=0.5, strip_zero=False)
  adjust = {"_left" : 0.0,
     "_right": 0.0,
     "_bottom" : -0.2,
     "_top":0.0,
     "_hspace":0.0}

  _fig, _subplot = plt.subplots(nrows=1, ncols=1, sharex=False)
  _fig.tight_layout()

  benchmarks = list(data['purecap'].keys())

  norm_evt = f"normalised-{event}"
  max_norm_err = 0.0
  max_ylim = 0.0
  for mode_idx, mode in enumerate(dev_modes): 
    mode_max_ylims = max([ plot_data[mode][_bm][norm_evt] for _bm in benchmarks ])
    if mode_max_ylims > max_ylim: 
      max_ylim = mode_max_ylims

    if event not in derived_events:
      norm_err = [ norm_conf_interval(data, mode, _bm, event, conf_interval) for _bm in benchmarks ]
      if max(norm_err) > max_norm_err: 
        max_norm_err = max(norm_err)

      # normalised performance for each event and each device mode 
      indv_norm_perf_count = [ plot_data[mode][_bm][norm_evt] \
                                        for _bm in benchmarks ]
      _subplot.bar( bar_x[mode_idx][:-1], indv_norm_perf_count,
                             label=march_y_labels[mode][0], color=m_arch_colors[mode], width=0.5, 
                             yerr=norm_err, capstyle='projecting', capsize=4)
      _subplot.bar( bar_x[mode_idx][-1],  [gmean(np.array(indv_norm_perf_count))],
                             color=m_arch_colors[mode], width=0.5)
    else: 
      # normalised performance for each event and each device mode 
      indv_norm_perf_count = [ plot_data[mode][_bm][norm_evt] \
                                        for _bm in benchmarks ]
      _subplot.bar( bar_x[mode_idx], indv_norm_perf_count + [gmean(np.array(indv_norm_perf_count))],
                             label=march_y_labels[mode][0], color=m_arch_colors[mode], width=0.5)
                             
  # Generate final parameters for each subplot
  _subplot.grid = True  
  _subplot.axvline(vline_barpos, 0,1, linestyle="--", color='k')
  _subplot.set_xticks(tick_x)
  _subplot.set_xticklabels( tick_lbl , rotation=15.0, fontsize="medium")
  _subplot.set_ylabel("Normalised (vs hybrid)", fontsize="medium")
  _subplot.set_ylim([None, math.ceil(max_ylim + max_norm_err)])
  _subplot.legend(loc='upper right', ncol=2, fontsize='medium')
  _subplot.set_title(norm_evt) 


def gen_combined_barchart(data, event_list, conf_interval=95):
  temp = dict(data)
  hybrid_data = temp.pop("hybrid") 
  plot_data = temp

  tick_lbl, tick_x, bar_x, vline_barpos = gen_bunched_bar_loc(plot_data, bw=0.5, sw=0.5, offset=0.5, strip_zero=False)
  adjust = {"_left" : 0.0,
     "_right": 0.0,
     "_bottom" : -0.2,
     "_top":0.0,
     "_hspace":0.0}

  _fig, _subplot = plt.subplots(nrows=len(event_list), ncols=1, sharex=False)
  _fig.tight_layout()

  benchmarks = list(data['purecap'].keys())
  event_counts = { _evt: {} for _evt in event_list } 
  for evt_idx, evt in enumerate(event_list): 
    norm_evt = f"normalised-{evt}"
    max_norm_err = 0.0
    max_ylim = 0.0
    for mode_idx, mode in enumerate(dev_modes): 
      mode_max_ylims = max([ plot_data[mode][_bm][norm_evt] for _bm in benchmarks ])
      if mode_max_ylims > max_ylim: 
        max_ylim = mode_max_ylims

      if evt not in derived_events:
        norm_err = [ norm_conf_interval(data, mode, _bm, evt, conf_interval) for _bm in benchmarks ]
        if max(norm_err) > max_norm_err: 
          max_norm_err = max(norm_err)
  
        # normalised performance for each event and each device mode 
        indv_norm_perf_count = [ plot_data[mode][_bm][norm_evt] \
                                          for _bm in benchmarks ]
        event_counts[evt][mode] = indv_norm_perf_count
        _subplot[evt_idx].bar( bar_x[mode_idx][:-1], event_counts[evt][mode],
                               label=march_y_labels[mode][0], color=m_arch_colors[mode], width=0.5, 
                               yerr=norm_err, capstyle='projecting', capsize=4)
        _subplot[evt_idx].bar( bar_x[mode_idx][-1],  [gmean(np.array(indv_norm_perf_count))],
                               color=m_arch_colors[mode], width=0.5)
      else:
        # normalised performance for each event and each device mode 
        indv_norm_perf_count = [ plot_data[mode][_bm][norm_evt] \
                                          for _bm in benchmarks ]
        event_counts[evt][mode] = indv_norm_perf_count
        _subplot[evt_idx].bar( bar_x[mode_idx], event_counts[evt][mode] + [gmean(np.array(indv_norm_perf_count))], 
                               label=march_y_labels[mode][0], color=m_arch_colors[mode], width=0.5 )

    # Generate final parameters for each subplot
    _subplot[evt_idx].grid = True  
    _subplot[evt_idx].axvline(vline_barpos, 0,1, linestyle="--", color='k')
    _subplot[evt_idx].set_xticks(tick_x)
    _subplot[evt_idx].set_xticklabels( tick_lbl , rotation=15.0, fontsize="medium")
    _subplot[evt_idx].set_ylabel("Normalised (vs hybrid)", fontsize="medium")
    _subplot[evt_idx].set_ylim([None, math.ceil(max_ylim + max_norm_err)])
    _subplot[evt_idx].legend(loc='upper right', ncol=2, fontsize='medium')
    _subplot[evt_idx].set_title(norm_evt) 


  
def plot(json_file, out_file, events, separate_files, conf_interval=95): 
  json_file = pathlib.Path(json_file)  # Ensure conversion to pathlib
  #assert plot_type == f"histogram", f"Only histograms are currently supported"
  result_data = None
  with open(json_file, "r") as fd:
    result_data = json.load(fd)

  assert len(events) > 0, f"Provide atleast one event to plot graph for" 

  if separate_files: 
    for event in events: 
      gen_barchart(result_data, event, conf_interval=conf_interval) 
      render(out_file.parent.resolve()/f"{out_file.stem}_{event}{out_file.suffix}")
  else: 
    gen_combined_barchart(result_data, events, conf_interval=conf_interval) 
    render(out_file)


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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog = sys.argv[0], 
                          description = f"Generate barcharts from json results file") 
  parser.add_argument('-i', '--infile',
                      required = True, 
                      help=f"input-data file name")
  parser.add_argument('-o', '--outfile',
                      default = 'output.pdf',
                       help=f"output-plot file name. suffix should end in {out_formats}")
  parser.add_argument('-t', '--graphtypes', nargs='*',
                             choices = ['individual', 'combined'], 
                             default = ['individual'],
                             help=f"graph types to plot execute. These may be combined together to do both")
                             #choices = ['individual', 'combined', 'separated', 'alloc-times'],
  parser.add_argument('-e', '--events', nargs='*',
                             choices = ['total-time', 'rss-kb', 'MEM_ACCESS', 'L2D_CACHE', 'L1D_CACHE', \
                                        'L1I_CACHE', 'INST_RETIRED', 'CPU_CYCLES', 'BUS_ACCESS', \
                                        'L1I_CACHE_REFILL', 'L1D_CACHE_REFILL', 'L2D_CACHE_REFILL', \
                                        'L1I_CACHE_', 'INST_RETIRED', 'CPU_CYCLES', 'BUS_ACCESS', \
                                        'LL_CACHE_RD', 'LL_CACHE_MISS_RD', 'BUS_ACCESS_RD_CTAG', \
                                        'gc-cycles', 'gc-time', 'gc-load'],
                             default = ['total-time'],
                             help=f"graph types to plot execute. These may be combined together to do both")


  args = parser.parse_args() 
  _input, _output = (pathlib.Path(args.infile), pathlib.Path(args.outfile).resolve())
  assert _input.exists(), f"Input file {_input} does not exist"
  assert _input.is_file(), f"Input {_input} is not a file"

  # Render individual performance parameters  separately
  if 'individual' in args.graphtypes:
    plot(_input, _output, args.events, True)

  plot(_input, _output, args.events, False)

#  for graph_type in args.graphtypes: 
#    if graph_type not in ['combined', 'separated']: 
#      continue
#
#    norm_graph = Combined_Graphs( "histogram", _input, _output)
#    try: 
#      graph_method = getattr(norm_graph, f"{graph_type}_normalised_plot") 
#    except AttributeError:
#      print(f"Could not find function {graph_type}_normalised_plot()")
#    else: 
#      graph_method((args.events, []), conf_interval=98)
#
#    #if graph_type == 'combined': 
#    #  norm_graph.combined_normalised_plot((args.events, []), conf_interval=98)
#    #elif graph_type == 'separated': 
#    #  norm_graph.separated_normalised_plot((args.events, []), conf_interval=98)
#
#  if 'alloc-times' in args.graphtypes:
#    alloc_times = GeoMean_Allocators( "histogram", _input, _output, ["total-time"])
#    alloc_times.normalised_alloc_plot(False)

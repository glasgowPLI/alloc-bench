
## QEMU command-line to execute
${HOME}/cheri/output/sdk/bin/qemu-system-morello -M virt,gic-version=3 -cpu morello \
  -bios edk2-aarch64-code.fd -m 2048 -nographic -drive \
  if=none,file=${HOME}/cheri/output/cheribsd-morello-purecap.img,id=drv,format=raw \
  -device virtio-blk-pci,drive=drv -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp::10005-:22 -device virtio-rng-pci  \
  -d plugin --plugin ${HOME}/cheri/build/qemu-build/contrib/plugins/libmemhowvec.so \
  -D /tmp/qemu-morello-tests/output-data/purecap_baseline.log \



### Hybrid 
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./binary_tree.elf && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./richards.elf && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./espresso.elf largest.espresso && cd ~ && shutdown -p now 
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./barnes.elf < input && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./cfrac.elf 17545186520507317056371138836327483792789528 && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./random_mixed_alloc.elf && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./small_fixed_alloc.elf && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./glibc_bench_simple.elf  && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./glibc_bench_thread.elf 4  && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./mstress.elf 4 50 25 && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid && LD_64_LIBRARY_PATH=$(pwd) ./xmalloc.elf -w 4 -t 16 -s 64  && cd ~  && shutdown -p now


### Hybrid NC 
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./binary_tree.elf && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./richards.elf && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./espresso.elf largest.espresso && cd ~ && shutdown -p now 
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./barnes.elf < input && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./cfrac.elf 17545186520507317056371138836327483792789528 && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./random_mixed_alloc.elf && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./small_fixed_alloc.elf && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./glibc_bench_simple.elf  && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./glibc_bench_thread.elf 4  && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./mstress.elf 4 50 25 && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_hybrid_nc && LD_64_LIBRARY_PATH=$(pwd) ./xmalloc.elf -w 4 -t 16 -s 64  && cd ~  && shutdown -p now

### Benchmark ABI 
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./binary_tree.elf && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./richards.elf && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./espresso.elf largest.espresso && cd ~ && shutdown -p now 
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./barnes.elf < input && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./cfrac.elf 17545186520507317056371138836327483792789528 && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./random_mixed_alloc.elf && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./small_fixed_alloc.elf && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./glibc_bench_simple.elf  && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./glibc_bench_thread.elf 4  && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./mstress.elf 4 50 25 && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_benchmarkabi && LD_64CB_LIBRARY_PATH=$(pwd) ./xmalloc.elf -w 4 -t 16 -s 64  && cd ~  && shutdown -p now


## Purecap 
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./binary_tree.elf && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./richards.elf && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./espresso.elf largest.espresso && cd ~ && shutdown -p now 
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./barnes.elf < input && cd ~ && shutdown -p now
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./cfrac.elf 17545186520507317056371138836327483792789528 && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./random_mixed_alloc.elf && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./small_fixed_alloc.elf && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./glibc_bench_simple.elf  && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./glibc_bench_thread.elf 4  && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./mstress.elf 4 50 25 && cd ~  && shutdown -p now
cd test-qemu/bench_bdwgc_purecap && LD_LIBRARY_PATH=$(pwd) ./xmalloc.elf -w 4 -t 16 -s 64  && cd ~  && shutdown -p now


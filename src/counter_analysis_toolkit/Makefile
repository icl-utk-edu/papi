USEMPI?=false
LDFLAGS=-L${PAPI_DIR}/lib -lpapi -lm -lpthread -ldl -lrt
INCFLAGS=-I${PAPI_DIR}/include
CFLAGS+=-g -Wall -Wextra
OPT0=-O0
OPT1=-O1
OPT2=-O2
OPT3=-O3
CC=gcc

## Check to see if MPI is used.
ifeq ($(USEMPI),true)
    CC=mpicc
    CFLAGS+=-DUSE_MPI
endif

## Try to auto-detect architecture if it is not set.
ifndef ARCH
    UNAME_M:=$(shell uname -m)

    ARCH=POWER
    ifeq ($(UNAME_M),x86_64)
        ARCH=X86
    endif
    ifeq ($(UNAME_M),aarch64)
        ARCH=ARM
    endif

    $(info    Detected ARCH=$(ARCH))
endif

## Architecture determines vector instruction set.
# Note, we can't use -mavx512fp16 -mavx512bf16 for ANY AVX unless AVX512 is avail.
ifeq ($(ARCH),X86)
    FLOP+=-mfma -DX86
    VECSRC=vec_dot_bf16 vec_fma_fp16 vec_fma_sp vec_fma_dp vec_nonfma_fp16 vec_nonfma_sp vec_nonfma_dp
    VEC_META=-DAVX128_AVAIL -DAVX256_AVAIL -DAVX512_AVAIL -DAVX512_FP16_AVAIL -DAVX512_BF16_AVAIL
    VEC128=-mavx -mavx512vl -mavx512fp16 -mavx512bf16 -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_128B
    VEC256=-mavx -mavx512vl -mavx512fp16 -mavx512bf16 -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_256B
    VEC512=-mavx -mavx512vl -mavx512fp16 -mavx512bf16 -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_512B
    VEC128_FMA=-mfma4 -mfma -mavx512vl -mavx512fp16 -mavx512bf16 -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_128B
    VEC256_FMA=-mfma4 -mfma -mavx512vl -mavx512fp16 -mavx512bf16 -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_256B
    VEC512_FMA=-mfma4 -mfma -mavx512vl -mavx512fp16 -mavx512bf16 -O0 -DX86 $(VEC_META) -DX86_VEC_WIDTH_512B
    VEC_ALL=-mavx -mavx512fp16 -mavx512bf16 -mfma4 -mfma -mavx512f -mavx512fp16 -mavx512bf16 -O0 -DX86 $(VEC_META)
    INSTR=-mavx512vl
endif
ifeq ($(ARCH),POWER)
    FLOP+=-maltivec -DPOWER
    VECSRC=vec_fma_fp16.o vec_fma_sp.o vec_fma_dp.o vec_nonfma_fp16.o vec_nonfma_sp.o vec_nonfma_dp.o
    VEC_ALL=-maltivec -DPOWER
endif
ifeq ($(ARCH),ARM)
    VEC_META=-DBF16_AVAIL -DFP16_AVAIL
    FLOP+=-march=armv8.2-a+fp16+bf16+sve -DARM $(VEC_META)
    VECSRC=vec_dot_bf16.o vec_fma_fp16.o vec_fma_sp.o vec_fma_dp.o vec_nonfma_fp16.o vec_nonfma_sp.o vec_nonfma_dp.o
    VEC_ALL=-march=armv8.2-a+fp16+bf16+sve -O0 -DARM $(VEC_META)
endif

all: branch.o d_cache eventstock.o flops i_cache instr vector
	make cat_collect

d_cache: timing_kernels.o prepareArray.o compar.o dcache.o

i_cache: icache.o icache_seq_kernel.o

vector: weak_symbols.o vec.o vec_scalar_verify.o $(VECSRC)

branch.o: branch.c branch.h
	$(CC) $(OPT0) $(CFLAGS) $(INCFLAGS) -c branch.c -o branch.o

timing_kernels.o: timing_kernels.c timing_kernels.h
	$(CC) $(OPT2) $(CFLAGS) -fopenmp $(INCFLAGS) -c timing_kernels.c -o timing_kernels.o

prepareArray.o: prepareArray.c prepareArray.h
	$(CC) $(OPT2) $(CFLAGS) -c prepareArray.c -o prepareArray.o

compar.o: compar.c
	$(CC) $(CFLAGS) $(OPT2) -c compar.c -o compar.o

dcache.o: dcache.c dcache.h
	$(CC) $(CFLAGS) $(OPT2) -fopenmp $(INCFLAGS) -c dcache.c -o dcache.o

eventstock.o: eventstock.c eventstock.h
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -c eventstock.c -o eventstock.o

flops: flops.c flops.h cat_arch.h
	$(CC) $(CFLAGS) $(FLOP) $(OPT1) $(INCFLAGS) -c flops.c -o flops.o

icache.o: icache.c icache.h
	bash gen_seq_dlopen.sh ${PWD}
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -c icache.c -o icache.o

icache_seq_kernel.o: icache_seq.c icache_seq.h icache_seq_kernel.c
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -c icache_seq.c -o icache_seq.o
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -fPIC -c icache_seq_kernel.c -o icache_seq_kernel.o
	$(CC) $(CFLAGS) $(OPT0) -shared -o icache_seq_kernel_0.so icache_seq_kernel.o
	bash replicate.sh ${PWD}
	rm ${PWD}/icache_seq_kernel.o

instr: instructions.c instr.h
	-$(CC) -c $(CFLAGS) $(OPT2) -ftree-vectorize $(FLOP) $(INSTR) $(INCFLAGS) instructions.c -o instructions.o

weak_symbols.o: weak_symbols.c vec.h
	-$(CC) -c $(CFLAGS) weak_symbols.c

vec.o: vec.c vec.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) -D$(ARCH) $(VEC_META) vec.c

vec_scalar_verify.o: vec_scalar_verify.c vec_scalar_verify.h cat_arch.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_ALL) vec_scalar_verify.c

vec_fma_fp16.o: vec_fma_fp16.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_ALL) vec_fma_fp16.c

vec_fma_fp16: vec_fma_fp16.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128_FMA) vec_fma_fp16.c -o vec_fma_fp16-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256_FMA) vec_fma_fp16.c -o vec_fma_fp16-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512_FMA) vec_fma_fp16.c -o vec_fma_fp16-512B.o

vec_dot_bf16.o: vec_dot_bf16.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_ALL) vec_dot_bf16.c

vec_dot_bf16: vec_dot_bf16.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128_FMA) vec_dot_bf16.c -o vec_dot_bf16-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256_FMA) vec_dot_bf16.c -o vec_dot_bf16-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512_FMA) vec_dot_bf16.c -o vec_dot_bf16-512B.o

vec_fma_sp.o: vec_fma_sp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_ALL) vec_fma_sp.c

vec_fma_sp: vec_fma_sp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128_FMA) vec_fma_sp.c -o vec_fma_sp-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256_FMA) vec_fma_sp.c -o vec_fma_sp-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512_FMA) vec_fma_sp.c -o vec_fma_sp-512B.o

vec_fma_dp.o: vec_fma_dp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_ALL) vec_fma_dp.c

vec_fma_dp: vec_fma_dp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128_FMA) vec_fma_dp.c -o vec_fma_dp-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256_FMA) vec_fma_dp.c -o vec_fma_dp-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512_FMA) vec_fma_dp.c -o vec_fma_dp-512B.o

vec_nonfma_fp16.o: vec_nonfma_fp16.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_ALL) vec_nonfma_fp16.c

vec_nonfma_fp16: vec_nonfma_fp16.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128) vec_nonfma_fp16.c -o vec_nonfma_fp16-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256) vec_nonfma_fp16.c -o vec_nonfma_fp16-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512) vec_nonfma_fp16.c -o vec_nonfma_fp16-512B.o

vec_nonfma_sp.o: vec_nonfma_sp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_ALL) vec_nonfma_sp.c

vec_nonfma_sp: vec_nonfma_sp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128) vec_nonfma_sp.c -o vec_nonfma_sp-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256) vec_nonfma_sp.c -o vec_nonfma_sp-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512) vec_nonfma_sp.c -o vec_nonfma_sp-512B.o

vec_nonfma_dp.o: vec_nonfma_dp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC_ALL) vec_nonfma_dp.c

vec_nonfma_dp: vec_nonfma_dp.c vec_scalar_verify.h
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC128) vec_nonfma_dp.c -o vec_nonfma_dp-128B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC256) vec_nonfma_dp.c -o vec_nonfma_dp-256B.o
	-$(CC) -c $(CFLAGS) $(OPT1) $(INCFLAGS) $(VEC512) vec_nonfma_dp.c -o vec_nonfma_dp-512B.o

cat_collect:
	$(CC) $(CFLAGS) -fopenmp $(INCFLAGS) main.c $(wildcard *.o) -o cat_collect $(LDFLAGS)

clean:
	rm -f *.o

realclean:
	rm -f cat_collect *.o *.so icache_seq.c icache_seq.h icache_seq_kernel.c


PAPIDIR?=/opt/papi
LDFLAGS=-L$(PAPIDIR)/lib -lpapi -lm -lpthread -ldl -lrt
INCFLAGS=-I$(PAPIDIR)/include
CFLAGS+=-g -Wall -Wextra
OPT0=-O0
OPT1=-O1
OPT2=-O2
CC=gcc

all: cat_collect

d_cache: timing_kernels.o prepareArray.o compar.o dcache.o

i_cache: icache.o icache_seq_kernel_0.o

branch.o: branch.c branch.h
	$(CC) $(OPT0) $(CFLAGS) $(INCFLAGS) -c branch.c -o branch.o

timing_kernels.o: timing_kernels.c timing_kernels.h
	$(CC) $(OPT2) $(CFLAGS) $(INCFLAGS) -c timing_kernels.c -o timing_kernels.o

prepareArray.o: prepareArray.c prepareArray.h
	$(CC) $(OPT2) $(CFLAGS) -c prepareArray.c -o prepareArray.o

compar.o: compar.c
	$(CC) $(CFLAGS) $(OPT2) -c compar.c -o compar.o

dcache.o: dcache.c dcache.h
	$(CC) $(CFLAGS) $(OPT2) $(INCFLAGS) -c dcache.c -o dcache.o

eventstock.o: eventstock.c eventstock.h
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -c eventstock.c -o eventstock.o

flops: flops_aux.c flops_aux.h flops.c flops.h
	$(CC) $(CFLAGS) $(OPT1) $(INCFLAGS) -c flops.c -o flops.o
	$(CC) $(CFLAGS) $(OPT1) $(INCFLAGS) -c flops_aux.c -o flops_aux.o

icache.o: icache.c icache.h
	bash gen_seq_dlopen.sh
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -c icache.c -o icache.o

icache_seq_kernel_0.o: icache_seq.c icache_seq.h
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -c icache_seq.c -o icache_seq.o
	$(CC) $(CFLAGS) $(OPT0) $(INCFLAGS) -fPIC -c icache_seq_kernel.c -o icache_seq_kernel_0.o
	$(CC) $(CFLAGS) $(OPT0) -shared -o icache_seq_kernel_0.so icache_seq_kernel_0.o
	bash replicate.sh

cat_collect: main.c branch.o d_cache eventstock.o flops i_cache
	$(CC) $(CFLAGS) $(INCFLAGS) main.c flops_aux.o flops.o icache_seq.o icache.o branch.o dcache.o timing_kernels.o prepareArray.o compar.o eventstock.o -o cat_collect $(LDFLAGS)

clean:
	rm -f *.o

realclean:
	rm -f cat_collect *.o *.so icache_seq.c icache_seq.h icache_seq_kernel.c


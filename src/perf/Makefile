all: libperf.a
	cd examples; $(MAKE)
perf.o: perf.c
	gcc -Wall -O2 -c perf.c -o perf.o
libperf.a: perf.o
	ar r libperf.a perf.o
clean:
	rm -f perf.o libperf.a *~
	cd examples; $(MAKE) clean
install: libperf.a
	install -o root -g root libperf.a /usr/lib
	install -o root -g root perf.h    /usr/include
	install -o root -g root perf.2    /usr/man/man2


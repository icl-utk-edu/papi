#
#  Test PAPI with fork and exec.
#

PAPI=/home/krentel/papi-2008.03.05/install

PAPI_INC_OPTS = -I $(PAPI)/include
PAPI_LD_OPTS =  -L $(PAPI)/lib -lpapi

CC = gcc
CFLAGS = -g -O -Wall

PROGRAMS = pfork pexec pchild

all: $(PROGRAMS)

cycles: cycles.c
	$(CC) -o $@ $(CFLAGS) $<

fork: fork.c
	$(CC) -o $@ $(CFLAGS) $<

callback.so: callback.c
	$(CC) -o $@ $(CFLAGS) -fPIC -shared $< $(PAPI_INC_OPTS) $(PAPI_LD_OPTS)

events: events.c
	$(CC) -o $@ $(CFLAGS) $< $(PAPI_INC_OPTS) $(PAPI_LD_OPTS)

pboth: pboth.c
	$(CC) -o $@ $(CFLAGS) $< $(PAPI_INC_OPTS) $(PAPI_LD_OPTS)

pfork: pfork.c
	$(CC) -o $@ $(CFLAGS) $< $(PAPI_INC_OPTS) $(PAPI_LD_OPTS)

pexec: pexec.c
	$(CC) -o $@ $(CFLAGS) $< $(PAPI_INC_OPTS) $(PAPI_LD_OPTS)

pchild: pchild.c
	$(CC) -o $@ $(CFLAGS) $< $(PAPI_INC_OPTS) $(PAPI_LD_OPTS)

clean:
	rm -f *.o *~ a.out $(PROGRAMS)


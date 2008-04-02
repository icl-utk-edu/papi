
PAPI_INC_OPTS = -I /projects/pkgs/papi/include
PAPI_LD_OPTS =  -L /projects/pkgs/papi/lib -lpapi

CC = gcc
CFLAGS = -g  -Wall

all: over fail

over: over.c
	$(CC) -o $@ $< $(CFLAGS) $(PAPI_INC_OPTS) $(PAPI_LD_OPTS)

fail: fail.c
	$(CC) -o $@ $< $(CFLAGS) $(PAPI_INC_OPTS) $(PAPI_LD_OPTS)

clean:
	rm -f *.o *~ a.out over fail


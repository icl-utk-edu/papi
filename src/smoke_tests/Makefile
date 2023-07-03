
EXECUTABLES = simple threads

CC ?= gcc

PAPIROOT ?= /path/to/papi/install/prefix
CFLAGS ?= -O0 -pthread -I$(PAPIROOT)/include
LIBS = -L$(PAPIROOT)/lib  -lm -ldl -lpapi -Wl,-rpath=$(PAPIROOT)/lib

all: $(EXECUTABLES)

clean:
	/bin/rm -f core *.o $(EXECUTABLES)

i.SUFFIXES: .c .o
	.c.o:
	    $(CC) $(CFLAGS) -c $*.c

simple: simple.o 
	$(CC) $(CFLAGS) -o simple simple.o $(LIBS)
threads: threads.o 
	$(CC) $(CFLAGS) -o threads threads.o $(LIBS)

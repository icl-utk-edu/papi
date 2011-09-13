INCLUDE = -I.
LIBRARY = -lpapi
PAPILIB = $(LIBRARY)
CC	= gcc
CC_R	= $(CC) -pthread
CFLAGS	= -g -O -Wall

UTILOBJS= do_loops.o test_utils.o dummy.o

all:	$(UTILOBJS)

do_loops.o: do_loops.c papi_test.h test_utils.h
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c do_loops.c

dummy.o: dummy.c
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c dummy.c

test_utils.o: test_utils.c
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c test_utils.c

clean:
	rm -f *.o *.stderr *.stdout core *~ $(ALL)

install: 

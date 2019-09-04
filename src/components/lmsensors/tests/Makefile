NAME=lmsensors
include ../../Makefile_comp_tests.target

TESTS = test_lmsensors_events 

lmsensors_tests: $(TESTS)


%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

test_lmsensors_events: test_lmsensors_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ test_lmsensors_events.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o


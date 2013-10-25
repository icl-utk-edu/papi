NAME=infiniband
include ../../Makefile_comp_tests
include ../Makefile.infiniband

INFINIBANDLIBS = -L$(INFINIBAND_IBMAD_DIR)/lib64 -L$(INFINIBAND_IBUMAD_DIR)/lib64 -libumad -libmad

TESTS = infiniband_list_events infiniband_values_by_code 

infiniband_tests: $(TESTS)


%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

infiniband_list_events: infiniband_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(INFINIBANDLIBS)

infiniband_values_by_code: infiniband_values_by_code.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(INFINIBANDLIBS)

clean:
	rm -f $(TESTS) *.o


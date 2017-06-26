NAME=infiniband_umad
include ../../Makefile_comp_tests.target
include ../Makefile.infiniband_umad

INFINIBANDLIBS = -L$(INFINIBAND_IBMAD_DIR)/lib64 -L$(INFINIBAND_IBUMAD_DIR)/lib64 -libumad -libmad

TESTS = infiniband_umad_list_events infiniband_umad_values_by_code

infiniband_umad_tests: $(TESTS)


%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

infiniband_umad_list_events: infiniband_umad_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(INFINIBANDLIBS)

infiniband_umad_values_by_code: infiniband_umad_values_by_code.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(INFINIBANDLIBS)

clean:
	rm -f $(TESTS) *.o


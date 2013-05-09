include ../../Makefile_comp_tests
include ../Makefile.infiniband

INFINIBANDLIBS = -L$(INFINIBAND_LIBDIR) -libumad -libmad

TESTS = infiniband_list_events infiniband_values_by_code 

infiniband_tests: $(TESTS)


%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

infiniband_list_events: infiniband_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(INFINIBANDLIBS)

infiniband_values_by_code: infiniband_values_by_code.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS) $(INFINIBANDLIBS)

install:
	@echo "Infiniband tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/infiniband/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/infiniband/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/infiniband/tests \;
	-chmod go+rx $(DATADIR)/infiniband/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/infiniband/tests \;

clean:
	rm -f $(TESTS) *.o


include ../../Makefile_comp_tests

TESTS = net_list_events net_values_by_code net_values_by_name

net_tests: $(TESTS)


%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

net_list_events: net_list_events.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ net_list_events.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

net_values_by_code: net_values_by_code.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ net_values_by_code.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

net_values_by_name: net_values_by_name.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ net_values_by_name.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

install:
	@echo "NET tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/net/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/net/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/net/tests \;
	-chmod go+rx $(DATADIR)/net/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/net/tests \;

clean:
	rm -f $(TESTS) *.o


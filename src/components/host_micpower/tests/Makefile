include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = host_micpower_basic 

micpower_tests: $(TESTS)

host_micpower_basic: host_micpower_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o host_micpower_basic host_micpower_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

install:
	@echo "Host_micpower tests being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/host_micpower/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/host_micpower/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/host_micpower/tests \;
	-chmod go+rx $(DATADIR)/host_micpower/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/host_micpower/tests \;

clean:
	rm -f $(TESTS) *.o






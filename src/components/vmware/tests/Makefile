include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = vmware_basic

vmware_tests: $(TESTS)

vmware_basic: vmware_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o vmware_basic vmware_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

install:
	@echo "VMWARE tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/vmware/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/vmware/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/vmware/tests \;
	-chmod go+rx $(DATADIR)/vmware/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/vmware/tests \;

clean:
	rm -f $(TESTS) *.o






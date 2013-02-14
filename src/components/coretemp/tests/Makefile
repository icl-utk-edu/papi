include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = coretemp_basic coretemp_pretty

coretemp_tests: $(TESTS)

coretemp_basic: coretemp_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o coretemp_basic coretemp_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

coretemp_pretty: coretemp_pretty.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o coretemp_pretty coretemp_pretty.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

install:
	@echo "CORETEMP tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/coretemp/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/coretemp/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/coretemp/tests \;
	-chmod go+rx $(DATADIR)/coretemp/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/coretemp/tests \;

clean:
	rm -f $(TESTS) *.o






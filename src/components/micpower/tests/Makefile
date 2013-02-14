include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = micpower_basic 

micpower_tests: $(TESTS)

micpower_basic: micpower_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o micpower_basic micpower_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

install:
	@echo "MICPOWER tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/micpower/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/micpower/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/micpower/tests \;
	-chmod go+rx $(DATADIR)/micpower/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/micpower/tests \;

clean:
	rm -f $(TESTS) *.o






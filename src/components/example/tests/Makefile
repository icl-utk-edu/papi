include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = example_basic example_multiple_components

example_tests: $(TESTS)

example_basic: example_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o example_basic example_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

example_multiple_components: example_multiple_components.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o example_multiple_components example_multiple_components.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

install:
	@echo "EXAMPLE tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/example/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/example/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/example/tests \;
	-chmod go+rx $(DATADIR)/example/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/example/tests \;

clean:
	rm -f $(TESTS) *.o






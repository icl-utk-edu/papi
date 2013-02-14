include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = mx_basic mx_elapsed

mx_tests: $(TESTS)

mx_basic: mx_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o mx_basic mx_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 


mx_elapsed: mx_elapsed.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o mx_elapsed mx_elapsed.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

install:
	@echo "MX tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/mx/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/mx/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/mx/tests \;
	-chmod go+rx $(DATADIR)/mx/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/mx/tests \;

clean:
	rm -f $(TESTS) *.o






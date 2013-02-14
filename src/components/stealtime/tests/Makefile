include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = stealtime_basic

stealtime_tests: $(TESTS)

stealtime_basic: stealtime_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o stealtime_basic stealtime_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

install:
	@echo "STEALTIME tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/stealtime/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/stealtime/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/stealtime/tests \;
	-chmod go+rx $(DATADIR)/stealtime/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/stealtime/tests \;

clean:
	rm -f $(TESTS) *.o






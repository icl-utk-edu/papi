include ../../Makefile_comp_tests

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = lustre_basic

lustre_tests: $(TESTS)

lustre_basic: lustre_basic.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o lustre_basic lustre_basic.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

install:
	@echo "LUSTRE tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/lustre/tests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/lustre/tests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/lustre/tests \;
	-chmod go+rx $(DATADIR)/lustre/tests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/lustre/tests \;

clean:
	rm -f $(TESTS) *.o






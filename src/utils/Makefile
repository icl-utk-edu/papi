TARGETS = avail mem_info cost clockres native_avail command_line event_chooser
UTILOBJS= do_loops.o dummy.o test_utils.o
INCLUDE = -I.. -I/usr/include -I../ctests
PAPILIB=../$(LIBRARY)

util: $(TARGETS)

event_chooser: event_chooser.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) event_chooser.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o event_chooser -lm

clockres: ../ctests/clockcore.c clockres.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) ../ctests/clockcore.c clockres.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres -lm

cost: cost.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) cost.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o cost -lm

command_line: command_line.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) command_line.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o command_line

mem_info: mem_info.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) mem_info.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o mem_info

avail: avail.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) avail.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o avail

native_avail: native_avail.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) native_avail.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o native_avail

test_utils.o: ../ctests/test_utils.c ../ctests/papi_test.h ../ctests/test_utils.h
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c ../ctests/test_utils.c

do_loops.o: ../ctests/do_loops.c ../ctests/papi_test.h ../ctests/test_utils.h
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c ../ctests/do_loops.c

dummy.o: ../ctests/dummy.c
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c ../ctests/dummy.c

clean:
	rm -f *.o *.stderr *.stdout core *~

clobber:
	rm -f $(TARGETS)

install: $(TARGETS)
	@if [ \"$(PREFIX)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make PREFIX=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(PREFIX)\""; 
	-mkdir -p $(PREFIX)/share/papi/utils
	-chmod go+rx $(PREFIX)/share/papi/utils
	-find . -perm -100 -type f -exec cp {} $(PREFIX)/share/papi/utils \;
	-chmod go+rx $(PREFIX)/share/papi/utils/*
	-find . -name "*.[ch]" -type f -exec cp {} $(PREFIX)/share/papi/utils \;

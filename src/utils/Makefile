# File: utils/Makefile
include Makefile.target

INCLUDE = -I../testlib -I.. -I.
testlibdir=../testlib
CLOCKCORE= $(testlibdir)/clockcore.o
DOLOOPS = $(testlibdir)/do_loops.o

ALL = papi_avail papi_mem_info papi_cost papi_clockres papi_native_avail \
	papi_command_line papi_event_chooser papi_decode papi_xml_event_info \
	papi_version papi_multiplex_cost papi_component_avail papi_error_codes

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c $<

default all utils: $(ALL)

papi_avail: papi_avail.o $(PAPILIB) print_header.o
	$(CC) $(LDFLAGS) -o papi_avail papi_avail.o print_header.o $(PAPILIB)

papi_clockres: papi_clockres.o $(PAPILIB) $(CLOCKCORE)
	$(CC) $(LDFLAGS) -o papi_clockres papi_clockres.o $(PAPILIB) $(CLOCKCORE) -lm

papi_command_line: papi_command_line.o $(PAPILIB) $(DOLOOPS)
	$(CC) $(LDFLAGS) -o papi_command_line papi_command_line.o $(PAPILIB) $(DOLOOPS)

papi_component_avail: papi_component_avail.o $(PAPILIB) print_header.o
	$(CC) $(LDFLAGS) -o papi_component_avail papi_component_avail.o $(PAPILIB) print_header.o

papi_cost: papi_cost.o $(PAPILIB) cost_utils.o
	$(CC) $(LDFLAGS) -o papi_cost papi_cost.o cost_utils.o $(PAPILIB) -lm

papi_decode: papi_decode.o $(PAPILIB)
	$(CC) $(LDFLAGS) -o papi_decode papi_decode.o $(PAPILIB)

papi_error_codes: papi_error_codes.o $(PAPILIB)
	$(CC) $(LDFLAGS) -o papi_error_codes papi_error_codes.o $(PAPILIB)

papi_event_chooser: papi_event_chooser.o $(PAPILIB) print_header.o
	$(CC) $(LDFLAGS) -o papi_event_chooser papi_event_chooser.o print_header.o $(PAPILIB)

papi_hybrid_native_avail: papi_hybrid_native_avail.o $(PAPILIB)
	$(CC) $(LDFLAGS) -o papi_hybrid_native_avail papi_hybrid_native_avail.o $(PAPILIB)

papi_mem_info: papi_mem_info.o $(PAPILIB)
	$(CC) $(LDFLAGS) -o papi_mem_info papi_mem_info.o $(PAPILIB)

papi_multiplex_cost: papi_multiplex_cost.o $(PAPILIB) cost_utils.o
	$(CC) $(LDFLAGS) -o papi_multiplex_cost papi_multiplex_cost.o cost_utils.o $(PAPILIB) -lm

papi_native_avail: papi_native_avail.o $(PAPILIB) print_header.o
	$(CC) $(LDFLAGS) -o papi_native_avail papi_native_avail.o $(PAPILIB) print_header.o

papi_version: papi_version.o $(PAPILIB)
	$(CC) $(LDFLAGS) -o papi_version papi_version.o $(PAPILIB)

papi_xml_event_info: papi_xml_event_info.o $(PAPILIB)
	$(CC) $(LDFLAGS) -o papi_xml_event_info papi_xml_event_info.o $(PAPILIB)



cost_utils.o: ../testlib/papi_test.h cost_utils.c
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) -c cost_utils.c

print_header.o: print_header.h print_header.c
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) -c print_header.c

clean:
	rm -f *.o *.stderr *.stdout core *~ $(ALL)

distclean clobber: clean
	rm -f Makefile.target

install: $(UTIL_TARGETS)
	@echo "Utilities (BINDIR) being installed in: \"$(BINDIR)\"";
	-mkdir -p $(BINDIR)
	-chmod go+rx $(BINDIR)
	-find . -perm -100 -type f -exec cp {} $(BINDIR) \;

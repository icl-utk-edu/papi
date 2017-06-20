# File: utils/Makefile
include Makefile.target

INCLUDE = -I../testlib -I.. -I.
PAPILIB = ../$(LIBRARY)
testlibdir=../testlib
TESTLIB= $(testlibdir)/libtestlib.a
CLOCKCORE= $(testlibdir)/clockcore.o

ALL = papi_avail papi_mem_info papi_cost papi_clockres papi_native_avail \
	papi_command_line papi_event_chooser papi_decode papi_xml_event_info \
	papi_version papi_multiplex_cost papi_component_avail papi_error_codes

default all utils: $(ALL)

papi_avail: papi_avail.c $(PAPILIB) print_header.o
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_avail.c $(PAPILIB) $(LDFLAGS) -o papi_avail print_header.o

papi_clockres: papi_clockres.c $(PAPILIB) $(CLOCKCORE)
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_clockres.c $(CLOCKCORE) $(PAPILIB) $(LDFLAGS) -o papi_clockres -lm

papi_component_avail: papi_component_avail.c $(PAPILIB) print_header.o
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_component_avail.c $(PAPILIB) print_header.o $(LDFLAGS) -o papi_component_avail

papi_decode: papi_decode.c $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_decode.c $(PAPILIB) $(LDFLAGS) -o papi_decode

papi_error_codes: papi_error_codes.c $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_error_codes.c $(PAPILIB) $(LDFLAGS) -o papi_error_codes

papi_event_chooser: papi_event_chooser.c $(PAPILIB) $(TESTLIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_event_chooser.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_event_chooser

papi_mem_info: papi_mem_info.c $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_mem_info.c $(PAPILIB) $(LDFLAGS) -o papi_mem_info


papi_native_avail: papi_native_avail.c $(PAPILIB) print_header.o
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_native_avail.c $(PAPILIB) print_header.o $(LDFLAGS) -o papi_native_avail

papi_version: papi_version.c $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_version.c $(PAPILIB) $(LDFLAGS) -o papi_version

papi_xml_event_info: papi_xml_event_info.c $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_xml_event_info.c $(PAPILIB) $(LDFLAGS) -o papi_xml_event_info



papi_cost: papi_cost.c $(TESTLIB) $(PAPILIB) cost_utils.o
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_cost.c $(TESTLIB) cost_utils.o $(PAPILIB) $(LDFLAGS) -o papi_cost -lm

papi_multiplex_cost: papi_multiplex_cost.c $(TESTLIB) $(PAPILIB) cost_utils.o
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_multiplex_cost.c $(TESTLIB) cost_utils.o $(PAPILIB) $(LDFLAGS) -o papi_multiplex_cost -lm

papi_command_line: papi_command_line.c $(TESTLIB) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_command_line.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_command_line

papi_hybrid_native_avail: papi_hybrid_native_avail.c $(PAPILIB) $(TESTLIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) papi_hybrid_native_avail.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_native_avail



cost_utils.o: ../testlib/papi_test.h cost_utils.c
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) -c cost_utils.c

print_header.o: print_header.h print_header.c
	$(CC) $(INCLUDE) $(CFLAGS) $(OPTFLAGS) -c print_header.c

clean:
	rm -f *.o *.stderr *.stdout core *~ $(ALL)

install: $(UTIL_TARGETS)
	@echo "Utilities (BINDIR) being installed in: \"$(BINDIR)\"";
	-mkdir -p $(BINDIR)
	-chmod go+rx $(BINDIR)
	-find . -perm -100 -type f -exec cp {} $(BINDIR) \;

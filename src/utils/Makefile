# File: utils/Makefile
# CVS:  $Id$
INCLUDE = -I../testlib -I. 
LIBRARY = -lpapi
PAPILIB = $(LIBRARY)
TESTLIB = ../testlib/libtestlib.a
CC	= gcc
CC_R	= $(CC) -pthread
CFLAGS	= -g -O -Wall

ALL = papi_avail papi_mem_info papi_cost papi_clockres papi_native_avail \
	papi_command_line papi_event_chooser papi_decode papi_xml_event_info \
	papi_version papi_multiplex_cost papi_component_avail papi_error_codes

default all utils: $(UTIL_TARGETS)

papi_event_chooser: event_chooser.c $(PAPILIB) $(TESTLIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) event_chooser.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_event_chooser -lm

papi_xml_event_info: event_info.c $(PAPILIB) $(TESTLIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) event_info.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_xml_event_info -lm

papi_clockres: clockres.c $(PAPILIB) $(TESTLIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) clockres.c $(TESTLIB) \
		$(PAPILIB) $(LDFLAGS) -o papi_clockres -lm

papi_cost: cost.c $(TESTLIB) $(PAPILIB) cost_utils.o
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) cost.c $(TESTLIB) cost_utils.o $(PAPILIB) $(LDFLAGS) -o papi_cost -lm

papi_multiplex_cost: multiplex_cost.c $(TESTLIB) $(PAPILIB) cost_utils.o
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) multiplex_cost.c $(TESTLIB) cost_utils.o $(PAPILIB) $(LDFLAGS) -o papi_multiplex_cost -lm

papi_command_line: command_line.c $(TESTLIB) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) command_line.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_command_line

papi_mem_info: mem_info.c $(PAPILIB) $(TESTLIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) mem_info.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_mem_info

papi_version: version.c $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) version.c $(PAPILIB) $(LDFLAGS) -o papi_version

papi_avail: avail.c $(PAPILIB) $(TESTLIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) avail.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_avail

papi_decode: decode.c $(PAPILIB) $(TESTLIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) decode.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_decode

papi_native_avail: native_avail.c $(PAPILIB) $(TESTLIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) native_avail.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_native_avail

papi_component_avail: component.c $(PAPILIB) $(TESTLIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) component.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_component_avail

papi_error_codes: error_codes.c $(PAPILIB) $(TESTLIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) error_codes.c $(TESTLIB) $(PAPILIB) $(LDFLAGS) -o papi_error_codes

cost_utils.o: ../testlib/papi_test.h cost_utils.c
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c cost_utils.c

clean:
	rm -f *.o *.stderr *.stdout core *~ $(ALL)

install: $(UTIL_TARGETS)
	@echo "Utilities (BINDIR) being installed in: \"$(BINDIR)\""; 
	-mkdir -p $(BINDIR)
	-chmod go+rx $(BINDIR)
	-find . -perm -100 -type f -exec cp {} $(BINDIR) \;

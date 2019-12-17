# $Id$

COMPSRCS += components/io/linux-io.c
COMPOBJS += linux-io.o

linux-io.o: components/io/linux-io.c components/io/linux-io.h $(HEADERS)
	$(CC) $(LIBCFLAGS) $(OPTFLAGS) -c components/io/linux-io.c -o $@


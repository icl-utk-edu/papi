# $Id$

SHELL=/bin/sh
CC=gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
INCLDIR=../linux/include
CPPFLAGS=-I$(INCLDIR)
AR=ar
ARFLAGS=ruv
RANLIB=ranlib
OBJS=libperfctr.o event_set.o

# Prevent 16-byte stack alignment crap in gcc-2.95.
CFLAGS += $(shell if $(CC) -mpreferred-stack-boundary=2 -S -o /dev/null -x c /dev/null > /dev/null 2>&1; then echo "-mpreferred-stack-boundary=2"; fi)

# this code does not need -fno-strict-aliasing

default:	libperfctr.a libperfctr.so

libperfctr.so: libperfctr.a
	$(CC) $(CFLAGS) $(CPPFLAGS) -shared -fPIC libperfctr.c event_set.c -o $@

libperfctr.a:	$(OBJS)
	$(AR) $(ARFLAGS) libperfctr.a $(OBJS)
	$(RANLIB) libperfctr.a

$(OBJS):	libperfctr.h $(INCLDIR)/linux/perfctr.h $(INCLDIR)/asm/perfctr.h
event_set.o:	event_codes.h

$(INCLDIR)/asm/perfctr.h:
	cd ..; make configure

event_codes.h:	../etc/perfctr-events.tab ../etc/mk-events-h.awk
	awk -f ../etc/mk-events-h.awk < ../etc/perfctr-events.tab > event_codes.h

distclean realclean:	clean

clean:
	rm -f libperfctr.a $(OBJS) event_codes.h

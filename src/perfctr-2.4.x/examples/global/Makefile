SHELL=/bin/sh
CC=gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
INCLDIR=../../linux/include
CPPFLAGS=-I$(INCLDIR) -I../../usr.lib

default:	global

global:	global.o ../../usr.lib/libperfctr.a

global.o:	$(INCLDIR)/asm/perfctr.h

distclean realclean:	clean

clean:
	rm -f global global.o core a.out

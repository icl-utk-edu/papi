SHELL=/bin/sh
CC=gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
INCLDIR=../../linux/include
CPPFLAGS=-I$(INCLDIR) -I../../usr.lib

default:	perfex

perfex:	perfex.o ../../usr.lib/libperfctr.a

perfex.o:	$(INCLDIR)/asm/perfctr.h

distclean realclean:	clean

clean:
	rm -f perfex perfex.o core a.out

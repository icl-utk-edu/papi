SHELL=/bin/sh
CC=gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
INCLDIR=../../linux/include
CPPFLAGS=-I$(INCLDIR) -I../../usr.lib

default:	signal

signal:	signal.o ../../usr.lib/libperfctr.a

signal.o:	$(INCLDIR)/asm/perfctr.h

distclean realclean:	clean

clean:
	rm -f signal signal.o core a.out

SHELL=/bin/sh
CC=gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
INCLDIR=../../linux/include
CPPFLAGS=-I$(INCLDIR) -I../../usr.lib

default:	self

self:	self.o ../../usr.lib/libperfctr.a

self.o:	$(INCLDIR)/asm/perfctr.h

distclean realclean:	clean

clean:
	rm -f self self.o core a.out

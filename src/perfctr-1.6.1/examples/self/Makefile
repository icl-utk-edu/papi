SHELL=/bin/sh
-include ../../config.out
CC=gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
CPPFLAGS=-I$(INCLDIR) -I../../usr.lib

default:	self

self:	self.o ../../usr.lib/libperfctr.o

self.o:	../../config.out

distclean realclean:	clean

clean:
	rm -f self self.o core a.out

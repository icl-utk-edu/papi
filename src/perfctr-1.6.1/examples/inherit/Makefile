CC=gcc
-include ../../config.out
CFLAGS=-O2 -fomit-frame-pointer -Wall
CPPFLAGS=-I$(INCLDIR) -I../../usr.lib

default:	inherit

inherit:	inherit.o ../../usr.lib/libperfctr.o

inherit.o:	../../config.out

distclean realclean:	clean

clean:
	rm -f inherit inherit.o core a.out

SHELL=/bin/sh
-include ../../config.out
CC=gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
CPPFLAGS=-I$(INCLDIR) -I../../usr.lib

default:	global

global:	global.o ../../usr.lib/libperfctr.o

global.o:	../../config.out

distclean realclean:	clean

clean:
	rm -f global global.o core a.out

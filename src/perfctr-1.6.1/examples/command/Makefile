CC=gcc
-include ../../config.out
CFLAGS=-O2 -fomit-frame-pointer -Wall
CPPFLAGS=-I$(INCLDIR) -I../../usr.lib

default:	command

command:	command.o ../../usr.lib/libperfctr.o

command.o:	../../config.out

distclean realclean:	clean

clean:
	rm -f command command.o core a.out

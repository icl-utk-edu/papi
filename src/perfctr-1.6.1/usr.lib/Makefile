# $Id$

SHELL=/bin/sh
-include ../config.out
CC=gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
CPPFLAGS=-I$(INCLDIR)

# Prevent 16-byte stack alignment crap in gcc-2.95.
CFLAGS += $(shell if $(CC) -mpreferred-stack-boundary=2 -S -o /dev/null -x c /dev/null > /dev/null 2>&1; then echo "-mpreferred-stack-boundary=2"; fi)

# this code does not need -fno-strict-aliasing

default:	libperfctr.o

libperfctr.o:	../config.out libperfctr.h libperfctr.c $(INCLDIR)/linux/perfctr.h $(INCLDIR)/asm-i386/perfctr.h

distclean realclean:	clean

clean:
	rm -f libperfctr.o

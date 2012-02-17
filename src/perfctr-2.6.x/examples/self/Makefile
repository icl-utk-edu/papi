# $Id: Makefile,v 1.6.2.1 2007/02/11 20:14:31 mikpe Exp $

SHELL=/bin/sh
ARCH := $(shell uname -m | sed -e s/i.86/i386/ -e s/sun4u/sparc64/ -e s/arm.*/arm/ -e s/sa110/arm/)
CC=gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
CPPFLAGS=-I../../linux/include -I../../usr.lib
HDEP=../../usr.lib/libperfctr.h ../../linux/include/linux/perfctr.h ../../linux/include/asm/perfctr.h arch.h
TARGET=self
i386_OBJS=x86.o
x86_64_OBJS=x86.o
ppc_OBJS=ppc.o
arm_OBJS=arm.o
ARCH_OBJS=$($(ARCH)_OBJS)
OBJS=self.o $(ARCH_OBJS)

default: $(TARGET)

$(TARGET): $(OBJS) ../../usr.lib/libperfctr.a

$(OBJS): $(HDEP)

install:

distclean realclean:	clean

clean:
	rm -f $(TARGET) $(OBJS) core a.out

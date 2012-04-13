# $Id: Makefile,v 1.8 2005/04/09 10:51:17 mikpe Exp $

SHELL=/bin/sh
ARCH := $(shell uname -m | sed -e s/i.86/i386/ -e s/sun4u/sparc64/ -e s/arm.*/arm/ -e s/sa110/arm/)
CC=$(CROSS_COMPILE)gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
CPPFLAGS=-I../../linux/include -I../../usr.lib
HDEP=../../usr.lib/libperfctr.h ../../linux/include/linux/perfctr.h ../../linux/include/asm/perfctr.h arch.h
TARGET=perfex
i386_OBJS=x86.o
x86_64_OBJS=x86.o
ppc_OBJS=ppc.o
ppc64_OBJS=ppc64.o
ARCH_OBJS=$($(ARCH)_OBJS)
OBJS=perfex.o $(ARCH_OBJS)

ifeq ($(ARCH),ppc64)
CPPFLAGS += -DPPC64
endif

default: $(TARGET)

$(TARGET): $(OBJS) ../../usr.lib/libperfctr.a

$(OBJS): $(HDEP)

install: $(TARGET)
	-mkdir -p $(BINDIR)
	cp $(TARGET) $(BINDIR)/

distclean realclean mrproper:	clean

clean:
	rm -f $(TARGET) $(OBJS) core a.out

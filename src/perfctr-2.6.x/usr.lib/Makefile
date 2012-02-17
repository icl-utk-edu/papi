# $Id: Makefile,v 1.26.2.46 2010/11/07 19:48:14 mikpe Exp $

SHELL=/bin/sh
ARCH := $(shell uname -m | sed -e s/i.86/i386/ -e s/sun4u/sparc64/ -e s/arm.*/arm/ -e s/sa110/arm/)
CC=gcc
CFLAGS=-O2 -fomit-frame-pointer -Wall
BUILD_INCLDIR=../linux/include
CPPFLAGS=-I$(BUILD_INCLDIR)
LD=ld
AR=ar
ARFLAGS=ruv
RANLIB=ranlib

i386_ESOBJS=event_set_x86.o event_set_amd.o event_set_centaur.o\
	event_set_p4.o event_set_p5.o event_set_p6.o
x86_64_ESOBJS=event_set_x86.o event_set_amd.o event_set_p4.o
ppc_ESOBJS=event_set_ppc.o
arm_ESOBJS=event_set_arm.o
EVENT_SET_OBJS=$($(ARCH)_ESOBJS)

i386_OBJS=x86.o
x86_64_OBJS=x86.o
ppc_OBJS=ppc.o
arm_OBJS=arm.o
ARCH_OBJS=$($(ARCH)_OBJS)

i386_H=x86.h
x86_64_H=x86.h
ppc_H=ppc.h
arm_H=arm.h
ARCH_H=$($(ARCH)_H)

AR_OBJS=global.o misc.o virtual.o marshal.o $(EVENT_SET_OBJS) $(ARCH_OBJS)

SO_OBJS=$(AR_OBJS:.o=.os)

# This is exceedingly ugly.
i386_ABIVER=5
x86_64_ABIVER=6
ppc_ABIVER=5
arm_ABIVER=0
SO_ABIVER=$($(ARCH)_ABIVER)

SO_LIBVER=2.6.42
SO_NAME=libperfctr.so.$(SO_ABIVER)
SO_LIB=libperfctr.so.$(SO_ABIVER).$(SO_LIBVER)

HDEP=libperfctr.h $(BUILD_INCLDIR)/linux/perfctr.h $(BUILD_INCLDIR)/asm/perfctr.h

i386_ASM_DIR=x86
x86_64_ASM_DIR=x86
ppc_ASM_DIR=powerpc
arm_ASM_DIR=arm
ARCH_ASM_DIR=asm-$($(ARCH)_ASM_DIR)

INSTALL_FILES=$(BUILD_INCLDIR)/$(ARCH_ASM_DIR)/perfctr.h $(BUILD_INCLDIR)/linux/perfctr.h\
	libperfctr.h libperfctr.a libperfctr.so perfctr_event_codes.h
CLEAN_FILES=$(AR_OBJS) $(SO_OBJS) libperfctr.a libperfctr.so gen-event-codes perfctr_event_codes.h marshal.c marshal.h libperfctr.o

# Prevent 16-byte stack alignment crap in gcc-2.95.
CFLAGS += $(shell if $(CC) -mpreferred-stack-boundary=2 -S -o /dev/null -x c /dev/null > /dev/null 2>&1; then echo "-mpreferred-stack-boundary=2"; fi)

.SUFFIXES: .os

%.os: %.c
	$(COMPILE.c) -fPIC $(OUTPUT_OPTION) $<

# this code does not need -fno-strict-aliasing

default:	$(INSTALL_FILES) libperfctr.o

libperfctr.a:	$(AR_OBJS)
	$(AR) $(ARFLAGS) libperfctr.a $(AR_OBJS)
	$(RANLIB) libperfctr.a

# not installed, only built as a workaround for PAPI's broken Makefiles
libperfctr.o:	$(AR_OBJS)
	$(LD) -r -o $@ $(AR_OBJS)

libperfctr.so:	$(SO_OBJS)
	$(CC) -shared -o $@ -Wl,-soname,$(SO_NAME) $(SO_OBJS)

$(AR_OBJS):	$(HDEP)
$(SO_OBJS):	$(HDEP)
$(EVENT_SET_OBJS):	event_set.h

$(BUILD_INCLDIR)/asm/perfctr.h:
	cd ..; make configure

misc.o virtual.o global.o:	marshal.h
misc.o virtual.o:	arch.h $(ARCH_H)
marshal.o:	marshal.c marshal.h
marshal.c:
	ln -s ../linux/drivers/perfctr/marshal.c .
marshal.h:
	ln -s ../linux/drivers/perfctr/marshal.h .

perfctr_event_codes.h:	gen-event-codes
	./gen-event-codes > $@

gen-event-codes:	gen-event-codes.c $(EVENT_SET_OBJS)
	$(LINK.c) $^ -o $@

install:	$(INSTALL_FILES)
	-mkdir -p $(INCLDIR) $(INCLDIR)/asm $(INCLDIR)/linux
	cp -f libperfctr.h $(INCLDIR)/
	cp perfctr_event_codes.h $(INCLDIR)/
	cp -f ../linux/include/$(ARCH_ASM_DIR)/perfctr.h $(INCLDIR)/asm/
	cp -f ../linux/include/linux/perfctr.h $(INCLDIR)/linux/
	-mkdir -p $(LIBDIR)
	cp libperfctr.a $(LIBDIR)/
	cp libperfctr.so $(LIBDIR)/$(SO_LIB)
	ln -sf $(SO_LIB) $(LIBDIR)/$(SO_NAME)
	ln -sf $(SO_NAME) $(LIBDIR)/libperfctr.so

distclean realclean:	clean

clean:
	rm -f $(CLEAN_FILES)

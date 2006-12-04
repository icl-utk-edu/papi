#
# Copyright (C) 2002 Hewlett-Packard Co
# Contributed by Stephane Eranian <eranian@hpl.hp.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.  
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# This file is part of libpfm, a performance monitoring support library for
# applications on Linux/ia64.
#
TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/..

include $(TOPDIR)/config.mk
include $(TOPDIR)/rules.mk

#
# Library version
#
VERSION=2
REVISION=0
AGE=0

#
# pfmusr.o is unsupported in this version
#
OBJS=pfmlib_common.o 

#
# list all library support modules
#
SUPP_OBJS = pfmlib_generic.o pfmlib_itanium.o pfmlib_itanium2.o

ifeq ($(CONFIG_PFMLIB_GENERIC),y)
CFLAGS += -DCONFIG_PFMLIB_GENERIC
OBJS   += pfmlib_generic.o
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM),y)
OBJS   += pfmlib_itanium.o
CFLAGS += -DCONFIG_PFMLIB_ITANIUM
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM2),y)
OBJS   += pfmlib_itanium2.o
CFLAGS += -DCONFIG_PFMLIB_ITANIUM2
endif

INCDIR=-I$(PFMINCDIR) -I.

SLIBPFM=libpfm.so.$(VERSION).$(REVISION).$(AGE)
TARGETS=libpfm.a $(SLIBPFM)

all: check_support $(TARGETS)


SOBJS=$(OBJS:.o=.lo)

$(OBJS) $(SOBJS): $(TOPDIR)/config.mk $(TOPDIR)/rules.mk Makefile
#
# XXX: try to compact this with rest of model specific info
#
ifeq ($(CONFIG_PFMLIB_ITANIUM),y)
pfmlib_itanium.o  : pfmlib_itanium.c itanium_events.h
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM2),y)
pfmlib_itanium2.o  : pfmlib_itanium2.c itanium2_events.h
endif

libpfm.a:  $(OBJS)
	$(RM) $@
	$(AR) cru $@ $(OBJS)

$(SLIBPFM):  $(SOBJS)
	$(CC) -shared -Wl,-soname -Wl,libpfm.so.$(VERSION) -o $@ $(SOBJS)
	ln -sf $@ libpfm.so.$(VERSION)
	ln -sf libpfm.so.$(VERSION) libpfm.so

clean:
	$(RM) -f $(OBJS) $(SOBJS) $(SUPP_OBJS)

distclean: clean
	$(RM) -f $(TARGETS) *.so*

depend:
	$(MKDEP) $(SRCS)

check_support:
	@echo $(OBJS)
	@echo -n checking support settings:
ifeq ($(CONFIG_PFMLIB_GENERIC)$(CONFIG_PFMLIB_ITANIUM)$(CONFIG_PFMLIB_ITANIUM2),nnn)
	@echo " missing support in library Makefile"
	@exit 1
endif
	@echo " done"

install: $(TARGETS)
	-mkdir -p $(DESTDIR)/lib
	$(INSTALL) -m 644 $(TARGETS) $(DESTDIR)/lib
	cd $(DESTDIR)/lib; ln -sf $(SLIBPFM) libpfm.so.$(VERSION) 
	cd $(DESTDIR)/lib; ln -sf libpfm.so.$(VERSION) libpfm.so
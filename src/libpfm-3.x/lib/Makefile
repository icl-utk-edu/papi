#
# Copyright (c) 2002-2004 Hewlett-Packard Development Company, L.P.
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
VERSION=3
REVISION=1
AGE=0

#
# Common files
#
SRCS=pfmlib_common.c pfmlib_os.c

CFLAGS+=-D_REENTRANT

#
# list all library support modules
#
SUPP_OBJS = pfmlib_generic_ia64.o pfmlib_itanium.o pfmlib_itanium2.o

ifeq ($(CONFIG_PFMLIB_GENERIC_IA64),y)
SRCS   += pfmlib_generic_ia64.c
CFLAGS += -DCONFIG_PFMLIB_GENERIC_IA64
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM),y)
SRCS   += pfmlib_itanium.c
CFLAGS += -DCONFIG_PFMLIB_ITANIUM
pfmlib_itanium.o : itanium_events.h
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM2),y)
SRCS   += pfmlib_itanium2.c
CFLAGS += -DCONFIG_PFMLIB_ITANIUM2
pfmlib_itanium2.o : itanium2_events.h
endif

INCDIR=-I$(PFMINCDIR) -I.

SLIBPFM=libpfm.so.$(VERSION).$(REVISION).$(AGE)
TARGETS=libpfm.a $(SLIBPFM)

OBJS=$(SRCS:.c=.o)
SOBJS=$(OBJS:.o=.lo)

all: check_support $(TARGETS)

$(OBJS) $(SOBJS): $(TOPDIR)/config.mk $(TOPDIR)/rules.mk Makefile

libpfm.a:  $(OBJS)
	$(RM) $@
	$(AR) cru $@ $(OBJS)

$(SLIBPFM):  $(SOBJS)
	$(CC) -shared -Wl,-soname -Wl,libpfm.so.$(VERSION) -o $@ $(SOBJS)
	ln -sf $@ libpfm.so.$(VERSION)

clean:
	$(RM) -f $(OBJS) $(SOBJS) $(SUPP_OBJS)

distclean: clean
	$(RM) -f $(TARGETS) *.so*

depend:
	$(MKDEP) $(INCDIR) $(CFLAGS) $(SRCS)

check_support:
	@echo $(OBJS)
	@echo -n checking support settings:
ifeq ($(CONFIG_PFMLIB_GENERIC_IA64)$(CONFIG_PFMLIB_ITANIUM)$(CONFIG_PFMLIB_ITANIUM2),nnn)
	@echo " missing support in library Makefile"
	@exit 1
endif
	@echo " done"

install: $(TARGETS)
	-mkdir -p $(DESTDIR)/lib
	$(INSTALL) -m 644 $(TARGETS) $(DESTDIR)/lib

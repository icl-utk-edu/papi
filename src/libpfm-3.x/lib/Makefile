#
# Copyright (c) 2002-2006 Hewlett-Packard Development Company, L.P.
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
TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/..

include $(TOPDIR)/config.mk
include $(TOPDIR)/rules.mk

#
# Library version
#
VERSION=3
REVISION=2
AGE=0

#
# Common files
#
SRCS=pfmlib_common.c pfmlib_os.c pfmlib_priv.c

CFLAGS+=-D_REENTRANT

#
# list all library support modules
#
ifeq ($(CONFIG_PFMLIB_GEN_IA64),y)
SRCS   += pfmlib_gen_ia64.c
CFLAGS += -DCONFIG_PFMLIB_GEN_IA64
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM),y)
SRCS   += pfmlib_itanium.c
CFLAGS += -DCONFIG_PFMLIB_ITANIUM
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM2),y)
SRCS   += pfmlib_itanium2.c
CFLAGS += -DCONFIG_PFMLIB_ITANIUM2
endif

ifeq ($(CONFIG_PFMLIB_MONTECITO),y)
SRCS   += pfmlib_montecito.c
CFLAGS += -DCONFIG_PFMLIB_MONTECITO
endif

ifeq ($(CONFIG_PFMLIB_AMD64),y)
SRCS   += pfmlib_amd64.c
CFLAGS += -DCONFIG_PFMLIB_AMD64
endif

ifeq ($(CONFIG_PFMLIB_I386_P6),y)
SRCS   += pfmlib_i386_p6.c
CFLAGS += -DCONFIG_PFMLIB_I386_P6
endif

ifeq ($(CONFIG_PFMLIB_GEN_IA32),y)
SRCS   += pfmlib_gen_ia32.c
CFLAGS += -DCONFIG_PFMLIB_GEN_IA32
endif

ifeq ($(CONFIG_PFMLIB_PENTIUM4),y)
SRCS   += pfmlib_pentium4.c
CFLAGS += -DCONFIG_PFMLIB_PENTIUM4
endif

ifeq ($(CONFIG_PFMLIB_COREDUO),y)
SRCS   += pfmlib_coreduo.c
CFLAGS += -DCONFIG_PFMLIB_COREDUO
endif

ifeq ($(CONFIG_PFMLIB_GEN_MIPS64),y)
SRCS   += pfmlib_gen_mips64.c
CFLAGS += -DCONFIG_PFMLIB_GEN_MIPS64
endif

INCDIR=-I$(PFMINCDIR) -I.
SLIBPFM=libpfm.so.$(VERSION).$(REVISION).$(AGE)
TARGETS=libpfm.a $(SLIBPFM)

OBJS=$(SRCS:.c=.o)
SOBJS=$(OBJS:.o=.lo)

all: $(TARGETS)

$(OBJS) $(SOBJS): $(TOPDIR)/config.mk $(TOPDIR)/rules.mk Makefile pfmlib_priv.h

pfmlib_itanium.o : itanium_events.h
pfmlib_itanium2.o : itanium2_events.h
pfmlib_montecito.o : montecito_events.h
pfmlib_amd64.o : amd64_events.h
pfmlib_i386_p6.o : i386_p6_events.h
pfmlib_pentium4.o : pentium4_events.h
pfmlib_gen_mips64.o : gen_mips64_events.h
pfmlib_coreduo.o : coreduo_events.h

libpfm.a:  $(OBJS)
	$(RM) $@
	$(AR) cru $@ $(OBJS)

$(SLIBPFM):  $(SOBJS)
	$(CC) -shared -Wl,-soname -Wl,libpfm.so.$(VERSION) -o $@ $(SOBJS)
	$(LN) -sf $@ libpfm.so.$(VERSION)

clean:
	$(RM) -f $(OBJS) $(SOBJS) $(SUPP_OBJS)

distclean: clean
	$(RM) -f $(TARGETS) *.so* *~

depend:
	$(MKDEP) $(INCDIR) $(CFLAGS) $(SRCS)

install: $(TARGETS)
	-mkdir -p $(DESTLIBDIR)
	$(INSTALL) -m 644 $(TARGETS) $(DESTLIBDIR)

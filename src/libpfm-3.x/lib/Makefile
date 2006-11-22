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
INCDEP=$(INC_COMMON)

ifeq ($(ARCH),ia64)
INCDEP := $(INCDEP) $(INC_IA64)
endif

ifeq ($(ARCH),ia32)
INCDEP := $(INCDEP) $(INC_IA32)
endif

ifeq ($(ARCH),x86_64)
INCDEP := $(INCDEP) $(INC_X86_64)
endif

ifeq ($(ARCH),mips64)
INCDEP := $(INCDEP) $(INC_MIPS64)
endif

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

ifeq ($(CONFIG_PFMLIB_GEN_MIPS64),y)
SRCS   += pfmlib_gen_mips64.c
CFLAGS += -DCONFIG_PFMLIB_GEN_MIPS64
endif

CFLAGS+=-I.
ALIBPFM=libpfm.a
SLIBPFM=libpfm.so.$(VERSION).$(REVISION).$(AGE)
VLIBPFM=libpfm.so.$(VERSION)
TARGETS=$(ALIBPFM) $(SLIBPFM)

OBJS=$(SRCS:.c=.o)
SOBJS=$(OBJS:.o=.lo)

all: $(TARGETS)

$(OBJS) $(SOBJS): $(TOPDIR)/config.mk $(TOPDIR)/rules.mk Makefile $(INCDEP)

libpfm.a:  $(OBJS)
	$(RM) $@
	$(AR) cru $@ $(OBJS)

$(SLIBPFM):  $(SOBJS)
	$(CC) -shared -Wl,-soname -Wl,$(VLIBPFM) -o $@ $(SOBJS)
	$(LN) -sf $@ $(VLIBPFM)
	$(LN) -sf $@ libpfm.so

clean:
	$(RM) -f $(OBJS) $(SOBJS) $(SUPP_OBJS)

distclean: clean
	$(RM) -f $(TARGETS) *.so* *~

depend:
	$(MKDEP) $(CFLAGS) $(SRCS)

install: $(TARGETS)
	-mkdir -p $(DESTDIR)$(LIBDIR)
	$(INSTALL) -m 644 $(ALIBPFM) $(DESTDIR)$(LIBDIR)
	$(INSTALL) $(SLIBPFM) $(DESTDIR)$(LIBDIR)
	cd $(DESTDIR)$(LIBDIR); ln -sf $(SLIBPFM) $(VLIBPFM) 
	cd $(DESTDIR)$(LIBDIR); ln -sf $(SLIBPFM) libpfm.so

INC_COMMON= $(PFMINCDIR)/perfmon/pfmlib.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp.h \
  $(PFMINCDIR)/perfmon/pfmlib_os.h \
  $(PFMINCDIR)/perfmon/perfmon.h \
  $(PFMINCDIR)/perfmon/perfmon_dfl_smpl.h \
  pfmlib_priv.h pfmlib_priv_comp.h \

INC_IA64= $(PFMINCDIR)/perfmon/pfmlib_itanium.h \
  $(PFMINCDIR)/perfmon/pfmlib_itanium2.h \
  $(PFMINCDIR)/perfmon/pfmlib_montecito.h \
  $(PFMINCDIR)/perfmon/perfmon_compat.h \
  $(PFMINCDIR)/perfmon/perfmon_default_smpl.h \
  $(PFMINCDIR)/perfmon/perfmon_ia64.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_ia64.h \
  $(PFMINCDIR)/perfmon/pfmlib_gen_ia64.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_ia64.h \
  itanium_events.h  itanium2_events.h montecito_events.h

INC_X86_64= $(PFMINCDIR)/perfmon/perfmon_p4_pebs_smpl.h \
  $(PFMINCDIR)/perfmon/pfmlib_amd64.h \
  $(PFMINCDIR)/perfmon/pfmlib_gen_ia32.h \
  $(PFMINCDIR)/perfmon/pfmlib_pentium4.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_x86_64.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_x86_64.h \
  amd64_events.h  pentium4_events.h gen_ia32_events.h

INC_IA32= $(PFMINCDIR)/perfmon/pfmlib_pentium4.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_i386.h \
  $(PFMINCDIR)/perfmon/pfmlib_i386_p6.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_i386.h \
  $(PFMINCDIR)/perfmon/pfmlib_gen_ia32.h \
  amd64_events.h  i386_p6_events.h \
  pentium4_events.h gen_ia32_events.h coreduo_events.h

INC_MIPS64= $(PFMINCDIR)/perfmon/pfmlib_gen_mips64.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_mips64.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_mips64.h \
  gen_mips64_events.h

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

ifeq ($(CONFIG_PFMLIB_ARCH_IA64),y)
INCDEP := $(INCDEP) $(INC_IA64)
SRCS   += pfmlib_gen_ia64.c pfmlib_itanium.c pfmlib_itanium2.c pfmlib_montecito.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_IA64
endif

ifeq ($(CONFIG_PFMLIB_ARCH_I386),y)
INCDEP := $(INCDEP) $(INC_IA32)
SRCS   += pfmlib_i386_p6.c pfmlib_pentium4.c pfmlib_amd64.c pfmlib_core.c pfmlib_gen_ia32.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_I386
endif

ifeq ($(CONFIG_PFMLIB_ARCH_X86_64),y)
INCDEP := $(INCDEP) $(INC_X86_64)
SRCS   += pfmlib_pentium4.c pfmlib_amd64.c pfmlib_core.c pfmlib_gen_ia32.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_X86_64
endif

ifeq ($(CONFIG_PFMLIB_ARCH_MIPS64),y)
INCDEP := $(INCDEP) $(INC_MIPS64)
SRCS   += pfmlib_gen_mips64.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_MIPS64
endif

ifeq ($(CONFIG_PFMLIB_ARCH_POWERPC),y)
INCDEP := $(INCDEP) $(INC_POWERPC)
SRCS   += pfmlib_powerpc.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_POWERPC
endif

ifeq ($(CONFIG_PFMLIB_ARCH_CRAYX2),y)
INCDEP := $(INCDEP) $(INC_CRAYX2)
SRCS   += pfmlib_crayx2.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_CRAYX2
endif

ifeq ($(CONFIG_PFMLIB_ARCH_CRAYXT),y)
CFLAGS += -DCONFIG_PFMLIB_ARCH_CRAYXT
endif

CFLAGS+=-I.
ALIBPFM=libpfm.a

ifneq ($(CONFIG_PFMLIB_ARCH_CRAYX2),y)
SLIBPFM=libpfm.so.$(VERSION).$(REVISION).$(AGE)
VLIBPFM=libpfm.so.$(VERSION)
endif

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
	$(LN) $@ $(VLIBPFM)
	$(LN) $@ libpfm.so

clean:
	$(RM) -f *.o *.lo

distclean: clean
	$(RM) -f $(TARGETS) *.so* *~

depend:
	$(MKDEP) $(CFLAGS) $(SRCS)

install: $(TARGETS)
	-mkdir -p $(DESTDIR)$(LIBDIR)
	$(INSTALL) -m 644 $(ALIBPFM) $(DESTDIR)$(LIBDIR)
ifneq ($(CONFIG_PFMLIB_ARCH_CRAYX2),y)
	$(INSTALL) $(SLIBPFM) $(DESTDIR)$(LIBDIR)
	cd $(DESTDIR)$(LIBDIR); $(LN) $(SLIBPFM) $(VLIBPFM)
	cd $(DESTDIR)$(LIBDIR); $(LN) $(SLIBPFM) libpfm.so
endif

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

INC_X86_64= $(PFMINCDIR)/perfmon/perfmon_pebs_smpl.h \
  $(PFMINCDIR)/perfmon/pfmlib_amd64.h \
  $(PFMINCDIR)/perfmon/pfmlib_core.h \
  $(PFMINCDIR)/perfmon/pfmlib_gen_ia32.h \
  $(PFMINCDIR)/perfmon/pfmlib_pentium4.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_x86_64.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_x86_64.h \
  amd64_events.h  pentium4_events.h gen_ia32_events.h core_events.h

INC_IA32=$(PFMINCDIR)/perfmon/perfmon_pebs_smpl.h \
  $(PFMINCDIR)/perfmon/pfmlib_pentium4.h \
  $(PFMINCDIR)/perfmon/pfmlib_amd64.h \
  $(PFMINCDIR)/perfmon/pfmlib_core.h \
  $(PFMINCDIR)/perfmon/pfmlib_i386_p6.h \
  $(PFMINCDIR)/perfmon/pfmlib_gen_ia32.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_i386.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_i386.h \
  amd64_events.h  i386_p6_events.h \
  pentium4_events.h gen_ia32_events.h coreduo_events.h core_events.h

INC_MIPS64= $(PFMINCDIR)/perfmon/pfmlib_gen_mips64.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_mips64.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_mips64.h \
  gen_mips64_events.h

INC_POWERPC= $(PFMINCDIR)/perfmon/pfmlib_powerpc.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_powerpc.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_powerpc.h \
  power_events.h

INC_CRAYX2= $(PFMINCIDR)/perfmon/pfmlib_crayx2.h \
  crayx2_events.h pfmlib_crayx2_priv.h

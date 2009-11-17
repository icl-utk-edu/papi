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
# Common files
#
SRCS=pfmlib_common.c

ifeq ($(SYS),Linux)
SRCS += pfmlib_perf_event.c pfmlib_perf_event_pmu.c pfmlib_os_linux.c
endif

ifeq ($(SYS),Darwin)
SRCS += pfmlib_os_macos.c
endif

ifeq ($(SYS), SunOS)
SRCS += pfmlib_os_opensolaris.c
endif

CFLAGS+=-D_REENTRANT -I.

#
# list all library support modules
#
ifeq ($(CONFIG_PFMLIB_ARCH_IA64),y)
INCARCH = $(INC_IA64)
#SRCS   += pfmlib_gen_ia64.c pfmlib_itanium.c pfmlib_itanium2.c pfmlib_montecito.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_IA64
endif

ifeq ($(CONFIG_PFMLIB_ARCH_I386),y)
INCARCH = $(INC_IA32)
SRCS   += pfmlib_amd64.c pfmlib_intel_core.c pfmlib_intel_x86.c pfmlib_intel_x86_arch.c pfmlib_intel_atom.c \
	  pfmlib_intel_nhm_unc.c pfmlib_intel_coreduo.c pfmlib_intel_p6.c pfmlib_intel_nhm.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_I386
endif

ifeq ($(CONFIG_PFMLIB_ARCH_X86_64),y)
INCARCH = $(INC_X86_64)
SRCS   += pfmlib_amd64.c pfmlib_intel_core.c pfmlib_intel_x86.c pfmlib_intel_x86_arch.c pfmlib_intel_atom.c \
	  pfmlib_intel_nhm.c pfmlib_intel_nhm_unc.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_X86_64
endif

ifeq ($(CONFIG_PFMLIB_ARCH_MIPS64),y)
INCARCH = $(INC_MIPS64)
#SRCS   += pfmlib_gen_mips64.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_MIPS64

ifeq ($(CONFIG_PFMLIB_ARCH_SICORTEX),y)
  INCARCH = $(INC_SICORTEX)
  CFLAGS += -DCONFIG_PFMLIB_ARCH_SICORTEX
ifneq ($(SCINCDIR),)
  CFLAGS += -I$(SCINCDIR)
endif
  SRCS   += pfmlib_sicortex.c
endif
endif

ifeq ($(CONFIG_PFMLIB_ARCH_POWERPC),y)
INCARCH = $(INC_POWERPC)
SRCS   += pfmlib_gen_powerpc.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_POWERPC
endif

ifeq ($(CONFIG_PFMLIB_ARCH_SPARC),y)
INCARCH = $(INC_SPARC)
#SRCS   += pfmlib_sparc.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_SPARC
endif

ifeq ($(CONFIG_PFMLIB_ARCH_CRAYXT),y)
CFLAGS += -DCONFIG_PFMLIB_ARCH_CRAYXT
endif

ifeq ($(CONFIG_PFMLIB_CELL),y)
INCARCH = $(INC_CELL)
#SRCS   += pfmlib_cell.c
CFLAGS += -DCONFIG_PFMLIB_CELL
endif

ifeq ($(SYS),Linux)
SLDFLAGS=-shared -Wl,-soname -Wl,$(VLIBPFM)
SLIBPFM=libpfm.so.$(VERSION).$(REVISION).$(AGE)
VLIBPFM=libpfm.so.$(VERSION)
SOLIBEXT=so
endif

CFLAGS+=-I.
ALIBPFM=libpfm.a

TARGETS=$(ALIBPFM)

ifeq ($(CONFIG_PFMLIB_SHARED),y)
TARGETS += $(SLIBPFM)
endif

OBJS=$(SRCS:.c=.o)
SOBJS=$(OBJS:.o=.lo)

INC_COMMON= $(PFMINCDIR)/perfmon/pfmlib.h pfmlib_priv.h

INC_IA64=pfmlib_ia64_priv.h	 \
	events/itanium_events.h	 \
	events/itanium2_events.h \
	events/montecito_events.h

INC_IA32= pfmlib_intel_x86_priv.h	\
	events/amd64_events.h    	\
	events/intel_p6_events.h	\
	events/intel_netburst_events.h \
	events//intel_x86_arch_events.h \
	events/intel_coreduo_events.h  \
	events/intel_core_events.h	\
	events/intel_atom_events.h	\
	events/intel_nhm_events.h	\
	events/intel_nhm_unc_events.h

INC_X86_64=pfmlib_intel_x86_priv.h	\
	events/amd64_events.h    	\
	events/intel_netburst_events.h \
	events/intel_x86_arch_events.h \
	events/intel_core_events.h	\
	events/intel_atom_events.h	\
	events/intel_nhm_events.h	\
	events/intel_nhm_unc_events.h

INC_MIPS64=gen_mips64_events.h

INC_SICORTEX= $(INC_MIPS64)

INC_POWERPC=events/ppc970_events.h	\
	events/ppc970mp_events.h	\
	events/power4_events.h	\
	events/power5_events.h		\
	events/power5+_events.h	\
	events/power6_events.h		\
	events/powerpc_reg.h

INC_SPARC=events/ultra12_events.h	\
	events/ultra3_events.h		\
	events/ultra3plus_events.h	\
	events/ultra3i_events.h	\
	events/ultra4plus_events.h	\
	events/niagara1_events.h	\
	events/niagara2_events.h

INC_CELL=events/cell_events.h

INCDEP=$(INC_COMMON) $(INCARCH)

all: $(TARGETS)

$(OBJS) $(SOBJS): $(TOPDIR)/config.mk $(TOPDIR)/rules.mk Makefile $(INCDEP)

libpfm.a:  $(OBJS)
	$(RM) $@
	$(AR) cru $@ $(OBJS)

$(SLIBPFM):  $(SOBJS)
	$(CC) $(CFLAGS) $(SLDFLAGS)  -o $@ $(SOBJS)
	$(LN) $@ $(VLIBPFM)
	$(LN) $@ libpfm.$(SOLIBEXT)

clean:
	$(RM) -f *.o *.lo *.a *.so* *~ *.$(SOLIBEXT)

distclean: clean

depend:
	$(MKDEP) $(CFLAGS) $(SRCS)

install: $(TARGETS)

install:
	@echo building: $(TARGETS)
	-mkdir -p $(DESTDIR)$(LIBDIR)
	$(INSTALL) -m 644 $(ALIBPFM) $(DESTDIR)$(LIBDIR)
ifeq ($(CONFIG_PFMLIB_SHARED),y)
	$(INSTALL) $(SLIBPFM) $(DESTDIR)$(LIBDIR)
	cd $(DESTDIR)$(LIBDIR); $(LN) $(SLIBPFM) $(VLIBPFM)
	cd $(DESTDIR)$(LIBDIR); $(LN) $(SLIBPFM) libpfm.$(SOLIBEXT)
endif

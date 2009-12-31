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

.PHONY: all clean distclean depend

ifeq ($(CONFIG_PFMLIB_ARCH_IA64),y)
ARCH_MAN=
endif

ARCH_X86=libpfm_intel_core.3	\
	 libpfm_intel_x86_arch.3\
	 libpfm_amd64.3		\
	 libpfm_intel_atom.3	\
	 libpfm_intel_nhm.3	\
	 libpfm_intel_nhm_unc.3 \
	 libpfm_intel_p6.3	\
	 libpfm_intel_coreduo.3

ifeq ($(CONFIG_PFMLIB_ARCH_I386),y)
ARCH_MAN=$(ARCH_X86)
endif

ifeq ($(CONFIG_PFMLIB_ARCH_X86_64),y)
ARCH_MAN=$(ARCH_X86)
endif

ifeq ($(CONFIG_PFMLIB_ARCH_MIPS64),y)
endif

ifeq ($(CONFIG_PFMLIB_ARCH_POWERPC),y)
ARCH_MAN=
endif

ifeq ($(CONFIG_PFMLIB_ARCH_CRAYXT),y)
endif

ifeq ($(CONFIG_PFMLIB_CELL),y)
endif

GEN_MAN= libpfm.3 \
	pfm_find_event.3 \
	pfm_get_event_attr_info.3 \
	pfm_get_event_info.3 \
	pfm_get_event_encoding.3 \
	pfm_get_event_first.3 \
	pfm_get_event_next.3 \
	pfm_get_pmu_info.3 \
	pfm_get_nevents.3 \
	pfm_get_perf_event_encoding.3 \
	pfm_get_version.3 \
	pfm_initialize.3 \
	pfm_pmu_present.3 \
	pfm_strerror.3

MAN=$(GEN_MAN) $(ARCH_MAN)
install:

	-mkdir -p $(DESTDIR)$(MANDIR)/man3
	( cd man3; $(INSTALL) -m 644 $(MAN) $(DESTDIR)$(MANDIR)/man3 )

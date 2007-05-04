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

HEADERS=perfmon/perfmon.h          \
	perfmon/perfmon_dfl_smpl.h \
	perfmon/pfmlib.h           \
	perfmon/pfmlib_comp.h      \
	perfmon/pfmlib_os.h

ifeq ($(CONFIG_PFMLIB_ARCH_IA64),y)
HEADERS += perfmon/pfmlib_os_ia64.h       \
	   perfmon/pfmlib_comp_ia64.h     \
	   perfmon/perfmon_ia64.h         \
	   perfmon/perfmon_compat.h       \
	   perfmon/perfmon_default_smpl.h
endif

ifeq ($(CONFIG_PFMLIB_ARCH_X86_64),y)
HEADERS += perfmon/pfmlib_os_x86_64.h   \
	   perfmon/pfmlib_comp_x86_64.h \
	   perfmon/perfmon_pebs_smpl.h
endif

ifeq ($(CONFIG_PFMLIB_ARCH_I386),y)
HEADERS += perfmon/pfmlib_os_i386.h   \
	   perfmon/pfmlib_comp_i386.h \
	   perfmon/perfmon_pebs_smpl.h
endif

ifeq ($(CONFIG_PFMLIB_GENERIC_IA64),y)
HEADERS += perfmon/pfmlib_gen_ia64.h
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM),y)
HEADERS += perfmon/pfmlib_itanium.h
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM2),y)
HEADERS += perfmon/pfmlib_itanium2.h
endif

ifeq ($(CONFIG_PFMLIB_MONTECITO),y)
HEADERS += perfmon/pfmlib_montecito.h
endif

ifeq ($(CONFIG_PFMLIB_AMD64),y)
HEADERS += perfmon/pfmlib_amd64.h
endif

ifeq ($(CONFIG_PFMLIB_PENTIUM4),y)
HEADERS += perfmon/pfmlib_pentium4.h
endif

ifeq ($(CONFIG_PFMLIB_I386_P6),y)
HEADERS += perfmon/pfmlib_i386_p6.h
endif

ifeq ($(CONFIG_PFMLIB_CORE),y)
HEADERS += perfmon/pfmlib_core.h
endif

ifeq ($(CONFIG_PFMLIB_GEN_IA32),y)
HEADERS += perfmon/pfmlib_gen_ia32.h
endif

ifeq ($(CONFIG_PFMLIB_ARCH_MIPS64),y)
HEADERS += perfmon/pfmlib_os_mips64.h   \
	   perfmon/pfmlib_comp_mips64.h \
	   perfmon/pfmlib_gen_mips64.h
endif

.PHONY: all clean distclean depend

install:
	-mkdir -p $(DESTDIR)$(INCDIR)/perfmon
	$(INSTALL) -m 644 $(HEADERS) $(DESTDIR)$(INCDIR)/perfmon

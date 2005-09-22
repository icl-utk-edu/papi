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

SIMPLE_PAGES= \
libpfm.3 \
libpfm_itanium.3 \
libpfm_itanium2.3 \
pfm_dispatch_events.3 \
pfm_find_event.3 \
pfm_find_event_bycode.3 \
pfm_find_event_bycode_next.3 \
pfm_find_event_byname.3 \
pfm_force_pmu.3 \
pfm_get_event_code.3 \
pfm_get_event_counters.3 \
pfm_get_event_name.3 \
pfm_get_first_event.3 \
pfm_get_impl_counters.3 \
pfm_get_impl_pmcs.3 \
pfm_get_impl_pmds.3 \
pfm_get_next_event.3 \
pfm_get_num_counters.3 \
pfm_get_pmu_name.3 \
pfm_get_pmu_name_bytype.3 \
pfm_get_pmu_type.3 \
pfm_get_version.3 \
pfm_initialize.3 \
pfm_list_supported_pmus.3 \
pfm_pmu_is_supported.3 \
pfm_get_event_description.3 \
pfm_set_options.3 \
pfm_start.3 \
pfm_stop.3 \
pfm_strerror.3

MANDIR=$(DESTDIR)/share/man/man3

.PHONY: all clean distclean depend

install:
	-mkdir -p $(MANDIR)
	( cd man3; $(INSTALL) -m 644 $(SIMPLE_PAGES) $(MANDIR) )

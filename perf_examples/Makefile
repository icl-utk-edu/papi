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

DIRS=

ifeq ($(ARCH),ia64)
#DIRS +=ia64
endif

ifeq ($(ARCH),ia32)
DIRS +=x86
endif

ifeq ($(ARCH),x86_64)
DIRS +=x86
endif

ifeq ($(CONFIG_PFMLIB_ARCH_CRAYXT),y)
CFLAGS += -DCONFIG_PFMLIB_ARCH_CRAYXT
endif

CFLAGS+= -I. -D_GNU_SOURCE

ifeq ($(SYS),Linux)
CFLAGS+= -pthread
endif

PERF_EVENT_HDR=$(TOPDIR)/include/perfmon/perf_event.h

ifeq ($(SYS),Linux)
LPC_UTILS=perf_util.o
TARGETS+=self self_basic self_count task task_attach_timeout syst \
	notify_self notify_group task_smpl self_smpl_multi \
	self_pipe syst_count task_cpu

#XTRA += rtop
endif

EXAMPLESDIR=$(DOCDIR)/perf_examples

all: $(TARGETS) $(XTRA)
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done
#
# Many systems don't have ncurses-devel installed
#
rtop: rtop.o $(PFMLIB)
	-$(CC) $(CFLAGS) $(LDFLAGS) -D_GNU_SOURCE -o $@ $^ $(LIBS)  -lpthread -lncurses -lm

$(TARGETS):  %:%.o $(LPC_UTILS) $(PFMLIB) $(PERF_EVENT_HDR)
	$(CC) $(CFLAGS) -o $@ $(LDFLAGS) $^ $(LIBS) 

clean:
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done
	$(RM) -f *.o $(TARGETS) $(XTRA) *~

distclean: clean

install_examples: $(TARGETS)
	@echo installing: $(TARGETS)
	-mkdir -p $(DESTDIR)$(EXAMPLESDIR)
	$(INSTALL) -m 755 $(TARGETS) $(DESTDIR)$(EXAMPLESDIR)
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done
#
# examples are installed as part of the RPM install, typically in /usr/share/doc/libpfm-X.Y/
#
.PHONY: install depend install_examples

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
DIRS +=ia64
endif

ifeq ($(ARCH),ia32)
DIRS +=x86
endif

ifeq ($(ARCH),x86_64)
DIRS +=x86
endif

CFLAGS+= -pthread -D_GNU_SOURCE -I$(PFMINCDIR)
LIBS += -lm

TARGETS=self task task_attach task_attach_timeout syst \
	notify_self notify_self2 notify_self3 \
	whichpmu multiplex multiplex2 set_notify \
	showreginfo task_smpl task_smpl_user showevtinfo \
	pfmsetup

XTRA=self_standalone notify_standalone smpl_standalone rtop

all: $(TARGETS) $(XTRA)
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done

rtop: rtop.o detect_pmcs.o $(PFMLIB)
	$(CC) $(LDFLAGS) -D_GNU_SOURCE -o $@ $^ $(LIBS)  -lpthread -lncurses

self_standalone: self_standalone.o
	$(CC) $(LDFLAGS) -D_GNU_SOURCE -o $@ $^

notify_standalone: notify_standalone.o
	$(CC) $(LDFLAGS) -D_GNU_SOURCE -o $@ $^

smpl_standalone: smpl_standalone.o
	$(CC) $(LDFLAGS) -D_GNU_SOURCE -o $@ $^

$(TARGETS):  %:%.o  detect_pmcs.o $(PFMLIB)
	$(CC) $(CFLAGS) -o $@ $(LDFLAGS) $^ $(LIBS) 

clean:
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done
	$(RM) -f *.o 

smpl_standalone.o self_standalone.o notify_standalone.o : standalone.h

distclean: clean
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done
	$(RM) -f $(TARGETS) $(XTRA) *~

#
# examples are installed as part of the RPM install, typically in /usr/share/doc/libpfm-X.Y/
#
.PHONY: install depend

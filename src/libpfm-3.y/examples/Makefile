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

ifeq ($(CONFIG_PFMLIB_ARCH_CRAYXT),y)
CFLAGS += -DCONFIG_PFMLIB_ARCH_CRAYXT
endif

CFLAGS+= -I. -D_GNU_SOURCE
LIBS += -lm

ifeq ($(SYS),Linux)
CFLAGS+= -pthread
LIBS += -lrt
endif

TARGET_GEN=showevtinfo check_events

ifeq ($(SYS),Linux)
TARGET_LINUX +=self task task_attach task_attach_timeout syst \
	  notify_self notify_self2 notify_self3 \
	  multiplex multiplex2 set_notify whichpmu \
	  showreginfo task_smpl task_smpl_user \
	  pfmsetup self_smpl_multi self_pipe \
	  notify_self_fork

XTRA += rtop
endif

all: $(TARGET_GEN) $(TARGET_LINUX) $(XTRA)
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done
# Many systems don't have ncurses installed
rtop: rtop.o detect_pmcs.o $(PFMLIB)
	-$(CC) $(CFLAGS) $(LDFLAGS) -D_GNU_SOURCE -o $@ $^ $(LIBS)  -lpthread -lncurses

$(TARGET_LINUX):  %:%.o detect_pmcs.o $(PFMLIB)
	$(CC) $(CFLAGS) -o $@ $(LDFLAGS) $^ $(LIBS) 

$(TARGET_GEN):  %:%.o $(PFMLIB)
	$(CC) $(CFLAGS) -o $@ $(LDFLAGS) $^ $(LIBS) 
clean:
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done
	$(RM) -f *.o $(TARGET_LINUX) $(TARGET_GEN) $(XTRA) *~

distclean: clean

install_examples: $(TARGET_LINUX) $(TARGET_GEN)

install_examples:
	@echo installing: $(TARGET_LINUX) $(TARGET_GEN)
	-mkdir -p $(DESTDIR)$(EXAMPLESDIR)
	$(INSTALL) -m 755 $(TARGET_LINUX) $(TARGET_GEN) $(DESTDIR)$(EXAMPLESDIR)
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done
#
# examples are installed as part of the RPM install, typically in /usr/share/doc/libpfm-X.Y/
#

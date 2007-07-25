#
# Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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

TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/../..

include $(TOPDIR)/config.mk
include $(TOPDIR)/rules.mk

CFLAGS+= -pthread -D_GNU_SOURCE
LDFLAGS+=-static 

PFMSINCDIR=../include

#
# Library version
#
VERSION=0
REVISION=1
AGE=0

SRCS=libpfms.c

ifneq ($(CONFIG_PFMLIB_ARCH_CRAYX2),y)
SLIBPFM=libpfms.so.$(VERSION).$(REVISION).$(AGE)
endif
TARGETS=libpfms.a $(SLIBPFM)
OBJS=$(SRCS:.c=.o)
SOBJS=$(OBJS:.o=.lo)


CFLAGS+=-I$(PFMSINCDIR)

all: $(TARGETS)

$(OBJS) $(SOBJS): $(TOPDIR)/config.mk $(TOPDIR)/rules.mk Makefile

libpfms.a:  $(OBJS)
	$(RM) $@
	$(AR) cru $@ $(OBJS)

$(SLIBPFM):  $(SOBJS)
	$(CC) -shared -Wl,-soname -Wl,libpfms.so.$(VERSION) -o $@ $(SOBJS)
	$(LN) -sf $@ libpfms.so.$(VERSION)

clean:
	$(RM) -f $(OBJS) $(SOBJS) $(SUPP_OBJS)

distclean: clean
	$(RM) -f $(TARGETS) *.so* *~

.PHONY: install depend

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

include ../config.mk
include ../rules.mk

#LDFLAGS+=-static 
INCDIR=-I$(PFMINCDIR) 
LIBELF=-lelf
LIBS += -lpthread

SRCS=self.c task.c syst.c notify_self.c task_smpl.c task_dear.c whichpmu.c freesmpl.c showreset.c
TARGETS=self task syst notify_self whichpmu freesmpl showreset

ifeq ($(CONFIG_PFMLIB_ITANIUM),y)
SRCS +=ita_rr.c ita_irr.c ita_opcode.c ita_btb.c ita_dear.c
TARGETS +=ita_rr ita_irr ita_opcode ita_btb ita_dear
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM2),y)
SRCS +=ita2_opcode.c ita2_rr.c ita2_irr.c ita2_dear.c ita2_btb.c
TARGETS +=ita2_opcode ita2_rr ita2_irr ita2_dear ita2_btb
endif

PFMLIB=$(PFMLIBDIR)/libpfm.a

all: $(TARGETS)

$(TARGETS):  %:%.o $(PFMLIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ -lpfm $(LIBS)

clean:
	$(RM) -f *.o 

distclean: clean
	$(RM) -f $(TARGETS)

depend:
	$(MKDEP) $(SRCS)

#
# examples are installed as part of the RPM install, typically in /usr/share/doc/libpfm-X.Y/
#
.PHONY: install

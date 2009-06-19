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
# This file is part of libpfm, a performance monitoring support library for
# applications on Linux.
#
#
# IMPORTANT: compatibility examples are meant FOR IA-64 host ONLY!
#
TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/..

include ../config.mk
include ../rules.mk

CFLAGS += -pthread
LDFLAGS+=-static 

LIBS += -lm

INCDIR=-I$(PFMINCDIR) 

TARGETS=self task task_attach task_attach_timeout syst \
	notify_self notify_self2 notify_self3 \
	whichpmu showreset multiplex\
	task_smpl notify_self_fork

SRCS +=ita_rr.c ita_irr.c ita_opcode.c ita_btb.c ita_dear.c
TARGETS +=ita_rr ita_irr ita_opcode ita_btb ita_dear

SRCS +=ita2_opcode.c ita2_rr.c ita2_irr.c ita2_dear.c
TARGETS +=ita2_dear ita2_btb ita2_opcode ita2_rr ita2_irr

SRCS +=mont_opcode.c mont_rr.c mont_irr.c mont_dear.c mont_etb.c
TARGETS +=mont_dear mont_opcode mont_rr mont_irr mont_etb

PFMLIB=$(PFMLIBDIR)/libpfm.a

all: $(TARGETS)

notify_self2.o: notify_self2.c
	$(CC) $(INCDIR) $(CFLAGS) -D_GNU_SOURCE -c $*.c

$(TARGETS):  %:%.o $(PFMLIB)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(LIBS) 
clean:
	$(RM) -f *.o $(TARGETS) *~

distclean: clean

depend:
	$(MKDEP) $(INCDIR) $(CFLAGS) $(SRCS)

#
# examples are installed as part of the RPM install, typically in /usr/share/doc/libpfm-X.Y/
#

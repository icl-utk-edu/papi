#
# Copyright (C) 2001-2002 Hewlett-Packard Co
# Contributed by Stephane Eranian <eranian@hpl.hp.com>
#
# This file is part of pfmon, a sample tool to measure performance 
# of applications on Linux/ia64.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# 02111-1307 USA
#
TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/..

include ../config.mk
include ../rules.mk

LDFLAGS+=-static 

INCDIR=-I$(PFM_INC_DIR) 

SRCS=self.c task.c syst.c notify_self.c task_smpl.c task_dear.c whichpmu.c freesmpl.c showreset.c 
TARGETS=self task syst notify_self whichpmu freesmpl showreset 

ifeq ($(CONFIG_PFMLIB_ITANIUM),y)
SRCS +=ita_rr.c ita_irr.c ita_opcode.c ita_btb.c 
TARGETS +=ita_rr ita_irr ita_opcode ita_btb 
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM2),y)
SRCS +=ita2_opcode.c ita2_rr.c ita2_irr.c
TARGETS +=ita2_opcode ita2_rr ita2_irr
endif

all: $(TARGETS)

$(TARGETS):  %:%.o $(LIBDIR)/libpfm.a 
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^

clean:
	rm -f *.o $(TARGETS)

distclean: clean

depend:
	$(MKDEP) $(SRCS)

.PHONY: install


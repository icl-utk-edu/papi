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

#LDFLAGS+=-static 

INCDIR=-I$(PFM_INC_DIR) 

SRCS=self.c task.c syst.c notify_self.c task_smpl.c task_dear.c ita_rr.c whichcpu.c ita_opcode.c \
     ita_btb.c simple_ear.c freesmpl.c

TARGETS=simple_ear self task syst notify_self whichcpu ita_rr ita_opcode ita_btb \
        freesmpl

all: $(TARGETS)

$(TARGETS):  %:%.o $(LIBDIR)/libpfm.a 
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^

clean:
	rm -f *.o $(TARGETS)

distclean: clean

depend:
	$(MKDEP) $(SRCS)

.PHONY: install


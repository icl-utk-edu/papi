#
# Copyright (C) 2001 Hewlett-Packard Co
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

INCDIR=-I$(PFM_INC_DIR) -I.

TARGETS=noploop hans

all: $(TARGETS)

noploop:  %:%.o $(LIBDIR)/libpfm.a 
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^

hans:  %:%.o $(LIBDIR)/libpfm.a 
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^

clean:
	rm -f hans.o noploop.o $(TARGETS)

distclean: clean

depend:
	$(MKDEP) $(SRCS)

#
# Copyright (C) 2002 Hewlett-Packard Co
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

TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/../..

include $(TOPDIR)/config.mk
include $(TOPDIR)/rules.mk

INCDIR=-I$(PFM_INC_DIR) -I. -I..

SRCS=raw_smpl.c compact_smpl.c text_smpl.c detailed_ita_smpl.c \
     example_smpl.c
OBJS=raw_smpl.o compact_smpl.o detailed_ita_smpl.o\
     example_smpl.o

TARGET=libsmplfmt.a


all: $(TARGET)

$(TARGET):  $(OBJS)
	rm -f $@
	$(AR) rv $@ $(OBJS)

$(OBJS) : $(TOPDIR)/config.mk

clean:
	rm -f $(OBJS) $(TARGET)

distclean: clean

depend:
	$(MKDEP) $(SRCS)

.PHONY: install

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

include $(TOPDIR)/config.mk
include $(TOPDIR)/rules.mk

ifeq ($(CONFIG_NUE_HACK),y)
CFLAGS += -DNUE_HACK
endif

#
# pfmusr.o is unsupported in this version
#
OBJS=pfmlib_common.o 

#
# list all library support modules
#
SUPP_OBJS = pfmlib_generic.o pfmlib_itanium.o

#
# list all events databases
#
SUPP_DB=itanium_events.db
SUPP_DB_TARGETS=$(SUPP_DB:.db=.h)

ifeq ($(CONFIG_PFMLIB_GENERIC),y)
CFLAGS    += -DCONFIG_PFMLIB_GENERIC
OBJS += pfmlib_generic.o
# no event database for generic
endif

ifeq ($(CONFIG_PFMLIB_ITANIUM),y)
OBJS += pfmlib_itanium.o
CFLAGS    += -DCONFIG_PFMLIB_ITANIUM
DB        += itanium_events.db
endif

INCDIR=-I$(PFM_INC_DIR) -I.

TARGET=libpfm.a

.SUFFIXES: .db .h
.db.h:
	./genpme.py $*.db $@

all: check_support $(TARGET)

$(OBJS) : ../config.mk

#
# XXX: try to compact this with rest of model specific info
#
ifeq ($(CONFIG_PFMLIB_ITANIUM),y)
pfmlib_itanium.o  : pfmlib_itanium.c itanium_events.h
endif

$(TARGET):  $(OBJS)
	$(RM) $@
	$(AR) qv $@ $(OBJS)

#$(DB_TARGETS) : $(DB) genpme.py

clean:
	rm -f $(OBJS) $(SUPP_OBJS) $(TARGET)

distclean: clean
	rm -f $(SUPP_DB_TARGETS)

depend:
	$(MKDEP) $(SRCS)

check_support:
	@echo $(OBJS)
	@echo -n checking support settings:
ifeq ($(CONFIG_PFMLIB_GENERIC)$(CONFIG_PFMLIB_ITANIUM),nn)
	@echo " missing support in library Makefile"
	@exit 1
endif
	@echo " done"

install: $(TARGET)
	-mkdir -p $(EXEC_PREFIX)/lib
	$(INSTALL) -m 644 $(TARGET) $(EXEC_PREFIX)/lib


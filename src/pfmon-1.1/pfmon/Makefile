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

LDFLAGS += -static

LIBS += -lpthread -lelf
SRCS=pfmon.c pfmon_smpl.c pfmon_util.c pfmon_system.c pfmon_task.c pfmon_symbols.c pfmon_conf.c
OBJS=pfmon.o pfmon_util.o pfmon_smpl.o pfmon_system.o pfmon_task.o pfmon_symbols.o pfmon_conf.o

ALL_SUPPORT=pfmon_gen.o pfmon_itanium.o

CFLAGS += -DEXEC_PREFIX=\"$(EXEC_PREFIX)\"

ifeq ($(CONFIG_PFMON_GENERIC),y)
OBJS   += pfmon_generic.o
CFLAGS += -DCONFIG_PFMON_GENERIC
endif

ifeq ($(CONFIG_PFMON_ITANIUM),y)
OBJS   += pfmon_itanium.o
CFLAGS += -DCONFIG_PFMON_ITANIUM
endif

#
# say y to turn on debugging support with --debug option
#
PFMON_DEBUG=y

ifeq ($(PFMON_DEBUG),y)
CFLAGS += -DPFMON_DEBUG
endif


DIRS=smpl_fmt
SMPL_FMT_LIB=smpl_fmt/libsmplfmt.a

INCDIR=-I$(PFM_INC_DIR) -I.

TARGET=pfmon


all: subdirs check_support $(TARGET)

$(TARGET):  $(LIBDIR)/libpfm.a $(OBJS) $(SMPL_FMT_LIB)
	$(CC) $(OBJS) -o $@ $(CFLAGS) $(LDFLAGS) $(LIBDIR)/libpfm.a $(LIBS) $(SMPL_FMT_LIB)

$(OBJS) : $(TOPDIR)/config.mk

clean: clean_subdirs
	rm -f $(OBJS) $(TARGET) 

distclean: clean clean_subdirs
	rm -f $(ALL_SUPPORT)
depend:
	$(MKDEP) $(SRCS)

pfmon : $(OBJS) $(SUPPORT)

check_support:
ifeq ($(CONFIG_PFMON_GENERIC)$(CONFIG_PFMON_ITANIUM),nn)
	@echo " missing CPU specific support in pfmon Makefile"
	@exit 1
endif

install: $(TARGET)
	-mkdir -p $(EXEC_PREFIX)/bin
	$(INSTALL) -m 755 pfmon $(EXEC_PREFIX)/bin/pfmon
	-ln -s $(EXEC_PREFIX)/bin/pfmon $(EXEC_PREFIX)/bin/pfmon_gen

subdirs: 
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d all; done

clean_subdirs: 
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d clean; done


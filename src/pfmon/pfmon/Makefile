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


#
# Look in config.mk for options
#
TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)

DIRS=pfmon tests scripts

include config.mk

all: 
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done

clean: 
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done

distclean: 
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done

depend: 
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done

install: 
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done

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

#
# This file defines the global compilation settings.
# It is included by every Makefile
#
#

#
# Where should things go in the end. the package will put things in lib and
# bin under this base.
#
EXEC_PREFIX=/usr/local

#
# XXX: should split library and pfmon packages
#

# Configuration Paramaters for pfm library
CONFIG_PFMLIB_GENERIC=y
CONFIG_PFMLIB_ITANIUM=y

# Configuration Paramaters for pfmon
CONFIG_PFMON_GENERIC=y
CONFIG_PFMON_ITANIUM=y

OPTIM=-O2

#
# you shouldn't have to touch anything beyond this point
#
CFLAGS=$(OPTIM) -Wall -g $(CONFIG_FLAGS)
#LDFLAGS=-static
MKDEP=makedepend

#
# The entire package can be compiled using 
# ecc the Intel Itanium Compiler (beta 6.0)
#
CC=cc

LIBS=
INSTALL=install

PFM_INC_DIR=$(TOPDIR)/include
LIBDIR=$(TOPDIR)/libpfm


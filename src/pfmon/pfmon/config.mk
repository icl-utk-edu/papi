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
# DESTDIR : the root directory used to install the binaries into
# DESTROOT: the root directory once package is installed
#
# Both directories are different when source is built for inclusion into
# a package because typically a fake root is used
#
DESTDIR=/usr/local
DESTROOT=/usr/local

#
# The root directory where to find the perfmon header files and the library
#
# pfmon will look into $(LIBPFMDIR)/include and $(LIBPFMDIR)/lib
#
PFMROOTDIR=/usr

# Configuration Paramaters for pfmon
CONFIG_PFMON_GENERIC=y
CONFIG_PFMON_ITANIUM=y
CONFIG_PFMON_ITANIUM2=y

# configuration for pfmon's sampling output formats
# not all of them work with all PMU models
CONFIG_PFMON_SMPL_FMT_RAW=y
CONFIG_PFMON_SMPL_FMT_COMPACT=y
CONFIG_PFMON_SMPL_FMT_DET_ITA=y
CONFIG_PFMON_SMPL_FMT_BTB=y
CONFIG_PFMON_SMPL_FMT_EXAMPLE=y
CONFIG_PFMON_SMPL_FMT_DET_ITA2=y

#
# say y to enable pfmon debugging support
CONFIG_PFMON_DEBUG=y

OPTIM= -O2
LDFLAGS=
#
# you shouldn't have to touch anything beyond this point
#

#
# The entire package can be compiled using 
# ecc the Intel Itanium Compiler (beta 6.0)
#
#CC=/opt/gcc3.1/bin/gcc -Wall
#CC=ecc
CC=gcc -Wall

CFLAGS=$(OPTIM) -g $(CONFIG_FLAGS)
MKDEP=makedepend


LIBS=
INSTALL=install

PFMINCDIR=$(PFMROOTDIR)/include
PFMLIBDIR=$(PFMROOTDIR)/lib

#
# Copyright (C) 2001 Hewlett-Packard Co
# Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com>
#
#
# This file defines the global compilation settings.
# It is included by every Makefile
#
CFLAGS=-O2 -Wall -g
LDFLAGS=
MKDEP=makedepend
CC=gcc

#
# you shouldn't have to touch anything beyond this point
#
PFM_INC_DIR=$(TOPDIR)
LIBDIR=$(TOPDIR)


#!/bin/sh
# $Id$
# usage: etc/install.sh PREFIX BINDIR LIBDIR INCLDIR
# If unset, {BIN,LIB,INCL}DIR are given default values from PREFIX.
# Then make install2 is invoked with the final {BIN,LIB,INCL}DIR.

PREFIX=$1
BINDIR=$2
LIBDIR=$3
INCLDIR=$4

fix_var() {
    if [ -z "$1" ]; then
	if [ -z "$PREFIX" ]; then
	    echo Error: at least one of PREFIX and $2 must be given
	    exit 1
	fi
	eval "$2=$PREFIX/$3"
    fi
}

fix_var "$BINDIR" BINDIR bin
fix_var "$LIBDIR" LIBDIR lib
fix_var "$INCLDIR" INCLDIR include

exec make "BINDIR=$BINDIR" "LIBDIR=$LIBDIR" "INCLDIR=$INCLDIR" install2

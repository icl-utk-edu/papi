#!/bin/sh
# $Id: install.sh,v 1.2 2005/01/16 22:44:34 mikpe Exp $
# usage: etc/install.sh PREFIX BINDIR LIBDIR INCLDIR ARCH
# If unset, {BIN,LIB,INCL}DIR are given default values from PREFIX.
# Then make install2 is invoked with the final {BIN,LIB,INCL}DIR.

PREFIX=$1
BINDIR=$2
LIBDIR=$3
INCLDIR=$4
ARCH=$5

case "$ARCH" in
  x86_64)
    LIBSUFFIX=lib64
    ;;
  *)
    LIBSUFFIX=lib
    ;;
esac

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
fix_var "$LIBDIR" LIBDIR $LIBSUFFIX
fix_var "$INCLDIR" INCLDIR include

exec make "BINDIR=$BINDIR" "LIBDIR=$LIBDIR" "INCLDIR=$INCLDIR" install2

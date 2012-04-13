#!/bin/sh
# $Id: install.sh,v 1.1.2.2 2007/04/09 12:50:36 mikpe Exp $
# usage: etc/install.sh PREFIX BINDIR LIBDIR INCLDIR ETCDIR ARCH
# If unset, {BIN,LIB,INCL}DIR are given default values from PREFIX.
# Then make install2 is invoked with the final {BIN,LIB,INCL,ETC}DIR.

PREFIX=$1
BINDIR=$2
LIBDIR=$3
INCLDIR=$4
ETCDIR=$5
ARCH=$6

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

exec make "BINDIR=$BINDIR" "LIBDIR=$LIBDIR" "INCLDIR=$INCLDIR" "ETCDIR=$ETCDIR" install2

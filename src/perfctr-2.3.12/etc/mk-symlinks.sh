#!/bin/sh
# $Id$
# usage:
#	cd ${linuxsrcdir}
#	${perfctrdir}/etc/mk-symlinks.sh

progname=$0
perfctrdir=`echo ${progname} | sed 's+/etc/mk-symlinks.sh$++'`
echo ln -sf ${perfctrdir}/linux/include/asm-i386/perfctr.h include/asm-i386/
ln -sf ${perfctrdir}/linux/include/asm-i386/perfctr.h include/asm-i386/
echo ln -sf ${perfctrdir}/linux/include/linux/perfctr.h include/linux/
ln -sf ${perfctrdir}/linux/include/linux/perfctr.h include/linux/
echo ln -sf ${perfctrdir}/linux/drivers/perfctr drivers/
ln -sf ${perfctrdir}/linux/drivers/perfctr drivers/

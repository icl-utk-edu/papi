SHELL=/bin/sh
ARCH := $(shell uname -m | sed -e s/i.86/i386/ -e s/sun4u/sparc64/ -e s/arm.*/arm/ -e s/sa110/arm/)
ASM_ARCH := $(shell uname -m | sed -e s/i.86/i386/ -e s/sun4u/sparc64/ -e s/arm.*/arm/ -e s/sa110/arm/ -e s/ppc.*/powerpc/)
SUBDIRS=usr.lib examples

default:	linux/include/asm/perfctr.h
	$(MAKE) MAKETARGET=default all-subdirs

all-subdirs:
	-for dir in $(SUBDIRS); do \
		(cd $$dir; $(MAKE) $(MAKEARGS) $(MAKETARGET); cd ..); \
	done

linux/include/asm/perfctr.h config configure:
	rm -f linux/include/asm
	ln -s asm-${ASM_ARCH} linux/include/asm

install:	default
	etc/install.sh "$(PREFIX)" "$(BINDIR)" "$(LIBDIR)" "$(INCLDIR)" "$(ARCH)"

install2:
	$(MAKE) MAKEARGS="BINDIR=$(BINDIR) LIBDIR=$(LIBDIR) INCLDIR=$(INCLDIR)" MAKETARGET=install all-subdirs

distclean realclean mrproper:	clean
	rm -f linux/include/asm

linuxclean:
	rm -f linux/drivers/perfctr/*.mod.c
	rm -f linux/drivers/perfctr/*.ko
	rm -f linux/drivers/perfctr/*.o
	rm -f linux/drivers/perfctr/.*.flags
	rm -f linux/drivers/perfctr/.depend
	rm -f linux/drivers/perfctr/.*.cmd

clean:	linuxclean
	$(MAKE) MAKETARGET=clean all-subdirs

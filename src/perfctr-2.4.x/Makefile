SHELL=/bin/sh
SUBDIRS=usr.lib examples

default:	linux/include/asm/perfctr.h
	$(MAKE) MAKETARGET=default all-subdirs

all-subdirs:
	-for dir in $(SUBDIRS); do \
		(cd $$dir; $(MAKE) $(MAKETARGET); cd ..); \
	done

linux/include/asm/perfctr.h config configure:
	rm -f linux/include/asm
	ln -s asm-i386 linux/include/asm

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

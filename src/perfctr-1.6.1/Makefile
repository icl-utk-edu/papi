SHELL=/bin/sh
SUBDIRS=usr.lib examples

default:	config.out
	$(MAKE) MAKETARGET=default all-subdirs

all-subdirs:
	-for dir in $(SUBDIRS); do \
		(cd $$dir; $(MAKE) $(MAKETARGET); cd ..); \
	done

config.out:
	@echo 'you have to run ./configure first!' ; exit 1

config configure:
	@./configure

distclean realclean:	clean
	rm -f config.out

linuxclean:
	rm -f linux/drivers/perfctr/*.o
	rm -f linux/drivers/perfctr/.*.flags
	rm -f linux/drivers/perfctr/.depend

clean:	linuxclean
	$(MAKE) MAKETARGET=clean all-subdirs

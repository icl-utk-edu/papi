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

clean:
	$(MAKE) MAKETARGET=clean all-subdirs
	rm -f linux/drivers/perfctr/*.o
	rm -f linux/drivers/perfctr/.*.flags
	rm -f linux/drivers/perfctr/.depend

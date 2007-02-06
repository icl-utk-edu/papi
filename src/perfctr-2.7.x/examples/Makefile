SHELL=/bin/sh
SUBDIRS=self perfex signal

default:
	$(MAKE) MAKETARGET=default all-subdirs

all-subdirs:
	-for dir in $(SUBDIRS); do \
		(cd $$dir; $(MAKE) $(MAKEARGS) $(MAKETARGET); cd ..); \
	done

install:
	$(MAKE) MAKEARGS="BINDIR=$(BINDIR)" MAKETARGET=install all-subdirs

distclean realclean mrproper:
	$(MAKE) MAKETARGET=distclean all-subdirs

clean:
	$(MAKE) MAKETARGET=clean all-subdirs

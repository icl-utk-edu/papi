.PHONY:	clean doxygen-user doxygen-full install FORCE all

all: doxygen-user doxygen-html
	@echo "Built PAPI user documentation"

utils: FORCE
	doxygen Doxyfile.utils

doxygen-user: FORCE 
	doxygen Doxyfile.utils
	doxygen Doxyfile

doxygen-full: 
	doxygen Doxyfile.utils-everything
	doxygen Doxyfile-everything

doxygen-html: FORCE 
	doxygen Doxyfile.html

clean:
	rm -rf man html

install: doxygen-user
	@echo "Man pages (MANDIR) being installed in: \"$(MANDIR)\""; 
	-mkdir -p $(MANDIR)/man3
	-chmod go+rx $(MANDIR)/man3
	-cp man/man3/*.3 $(MANDIR)/man3
	-chmod go+r $(MANDIR)/man3/*.3
	-mkdir -p $(MANDIR)/man1
	-chmod go+rx $(MANDIR)/man1
	-cp man/man1/*.1 $(MANDIR)/man1
	-chmod go+r $(MANDIR)/man1/*.1

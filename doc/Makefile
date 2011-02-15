.PHONY:	clean doxygen-user doxygen-full install

doxygen-user:
	doxygen Doxyfile

doxygen-full:
	doxygen Doxyfile-everything

clean:
	rm -rf man html

install:
	@echo "Man pages (MANDIR) being installed in: \"$(MANDIR)\""; 
	-mkdir -p $(MANDIR)/man3
	-chmod go+rx $(MANDIR)/man3
	-cp man/man3/*.3 $(MANDIR)/man3
	-chmod go+r $(MANDIR)/man3/*.3
	#-mkdir -p $(MANDIR)/man1
	#-chmod go+rx $(MANDIR)/man1
	#-cp man/man1/*.1 $(MANDIR)/man1
	#-chmod go+r $(MANDIR)/man1/*.1

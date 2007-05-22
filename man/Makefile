clean:
	rm -f *~ core man3/*~

install: install-man install-doc

install-man:
	@echo "Man pages (MANDIR) being installed in: \"$(MANDIR)\""; 
	-mkdir -p $(MANDIR)/man3
	-chmod go+rx $(MANDIR)/man3
	-cp man3/PAPI*.3 $(MANDIR)/man3
	-chmod go+r $(MANDIR)/man3/PAPI*.3
	-mkdir -p $(MANDIR)/man1
	-chmod go+rx $(MANDIR)/man1
	-cp man1/*.1 $(MANDIR)/man1
	-chmod go+r $(MANDIR)/man1/*.1

install-doc:
	@echo "Docs (DOCDIR) being installed in: \"$(DOCDIR)\""; 
	-mkdir -p $(DOCDIR)/html
	-chmod go+rx $(DOCDIR)
	-cp ../*.txt ../doc/PAPI_*.doc ../doc/PAPI_*.pdf ../*.html $(DOCDIR)
	-chmod go+r $(DOCDIR)/*
	-cp ./html/*.html $(DOCDIR)/html
	-chmod go+r $(DOCDIR)/html/*


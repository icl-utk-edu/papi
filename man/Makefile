all:	man2html man2more

man2html:
	cc -g papiman2HTML.c -o man2html
man2more:
	cc -g papiman.c -o man2more

clean:
	rm -f man2more man2html *~ core man3/*~

install:
	@if [ \"$(DESTDIR)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make DESTDIR=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(DESTDIR)\""; 
	mkdir -p $(DESTDIR)/man/man3
	chmod go+rx $(DESTDIR)/man/man3
	cp man3/*.3 $(DESTDIR)/man/man3
	chmod go+r $(DESTDIR)/man/man3/*
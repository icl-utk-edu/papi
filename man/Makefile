clean:
	rm -f *~ core man3/*~

install:
	@if [ \"$(PREFIX)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make PREFIX=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(PREFIX)\""; 
	-mkdir -p $(PREFIX)/man/man3
	-chmod go+rx $(PREFIX)/man/man3
	-cp man3/PAPI*.3 $(PREFIX)/man/man3
	-chmod go+r $(PREFIX)/man/man3/PAPI*.3
	-mkdir -p $(PREFIX)/man/man1
	-chmod go+rx $(PREFIX)/man/man1
	-cp man1/PAPI*.1 $(PREFIX)/man/man1
	-chmod go+r $(PREFIX)/man/man1/PAPI*.1

.PHONY:	clean doxygen-user doxygen-full install all

all: doxygen-user 
	@echo "Built PAPI user documentation"

doxygen-user: 
	doxygen Doxyfile.utils
	doxygen Doxyfile

doxygen-full: 
	doxygen Doxyfile.utils
	doxygen Doxyfile-everything

clean:
	rm -rf man html

install: doxygen-user
	-rm -r ../man/man1/*.1 ../man/man3/*.3
	-cp -R man/man1/*.1 ../man/man1
	-cp -R man/man3/*.3 ../man/man3

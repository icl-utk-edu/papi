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
	-rm -rf ../man/man1 ../man/man3
	-cp -R man/ ../

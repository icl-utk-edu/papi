.PHONY:	clean install force_me all

all: man
	@echo "Built PAPI user documentation"

html: force_me
	doxygen Doxyfile-html

man: man/man1 man/man3

man/man3: ../src/papi.h 
	doxygen Doxyfile-man3

man/man1: ../src/utils/avail.c ../src/utils/clockres.c  ../src/utils/command_line.c ../src/utils/component.c ../src/utils/cost.c ../src/utils/decode.c ../src/utils/error_codes.c ../src/utils/event_chooser.c ../src/utils/event_info.c ../src/utils/mem_info.c ../src/utils/multiplex_cost.c ../src/utils/native_avail.c
	doxygen Doxyfile-man1
 
clean:
	rm -rf man html doxyerror

install: man
	-rm -r ../man/man1/*.1 ../man/man3/*.3
	-cp -R man/man1/*.1 ../man/man1
	-cp -R man/man3/*.3 ../man/man3

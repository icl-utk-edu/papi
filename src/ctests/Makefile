EXAMPLES = first second third fourth overflow profile
EXOBS    = first.o second.o third.o fourth.o overflow.o profile.o
CFLAGS 	 = -I.. -g

default:
	cd ..; $(MAKE)

tests: $(EXAMPLES)

$(EXAMPLES): %: %.o 
	$(CC) $< -o $@ -L.. -lpapi

clean:
	rm -f $(EXOBS) $(EXAMPLES) core *~

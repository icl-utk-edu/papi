EXAMPLES = first second third fourth overflow profile
EXOBS    = first.o second.o third.o fourth.o overflow.o profile.o

default:
	cd ..; $(MAKE)

tests: $(EXAMPLES)

$(EXAMPLES): %: %.o ../libpapi.a
	$(CC) $(LDFLAGS) $< -o $@ -lpapi

clean:
	rm -f $(EXOBS) $(EXAMPLES) core *~

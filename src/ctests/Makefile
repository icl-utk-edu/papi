EXAMPLES = first second third fourth overflow
EXOBS    = first.o second.o third.o fourth.o overflow.o

all: $(EXAMPLES)

$(EXAMPLES): %: %.o ../libpapi.a
	$(CC) $(LDFLAGS) $< -o $@ -lpapi

clean:
	rm -f $(EXOBS) $(EXAMPLES)

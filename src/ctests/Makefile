EXAMPLES = first third
EXOBS    = first.o third.o

all: $(EXAMPLES)

$(EXAMPLES): %: %.o ../libpapi.a
	$(CC) $(LDFLAGS) $< -o $@ -lpapi

clean:
	rm -f $(EXOBS) $(EXAMPLES)

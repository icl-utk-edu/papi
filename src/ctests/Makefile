EXAMPLES = first third x y f
EXOBS    = first.o third.o x.o y.o f.o

all: $(EXAMPLES)

$(EXAMPLES): %: %.o ../libpapi.a
	$(CC) $(LDFLAGS) $< -o $@ -lpapi

clean:
	rm -f $(EXOBS) $(EXAMPLES)

EXAMPLES = first 
EXOBS    = first.o

all: $(EXAMPLES)

$(EXAMPLES): %: %.o 
	$(CC) $(LDFLAGS) $< -o $@ -lpapi

clean:
	rm -f $(EXOBS) $(EXAMPLES)
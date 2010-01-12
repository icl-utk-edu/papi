TESTS   	= papi_1

bgp_tests: $(TESTS)

papi%:
	$(CC) $(INCLUDE) $(CFLAGS) -o $@ $@.c $(LIBRARY) $(LDFLAGS)

clean:
	rm -f core $(TESTS)  *~ *.o

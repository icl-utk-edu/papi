CC = gcc
CFLAGS = -O2 -Wall -fopenmp
LFLAGS = 
PAPI_INCLUDE = ../../.. 
PAPI_LIBRARY = ../../../libpapi.so

#all:	powercap_write_test

powercap_write_test:	powercap_write_test.o
	$(CC) $(LFLAGS) -o powercap_write_test -fopenmp powercap_write_test.o  $(PAPI_LIBRARY)

powercap_write_test.o:	powercap_write_test.c
	$(CC) $(CFLAGS) -I$(PAPI_INCLUDE)  -c powercap_write_test.c

clean:	
	rm -f *~ *.o powercap_write_test *.txt *.tsv *.dat *.jpg


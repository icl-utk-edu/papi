thread:
	gcc -g -Wall -c -I. libperfctr.c
	gcc -g -Wall -I../tests thread.c libperfctr.o ../tests/do_loops.c -lpthread -o thread

clean:
	rm -f *~ core thread *.o

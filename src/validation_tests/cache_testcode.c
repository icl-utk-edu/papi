#include <stdio.h>

int cache_write_test(double *array, int size) {
	int i;

	for(i=0; i<size; i++) {
		array[i]=(double)i;
	}

	return 0;
}

double cache_read_test(double *array, int size) {

	int i;
	double sum=0;

	for(i=0; i<size; i++) {
		sum+= array[i];
	}

	return sum;
}

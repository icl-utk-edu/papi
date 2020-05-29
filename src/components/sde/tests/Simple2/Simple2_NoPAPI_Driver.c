#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void simple_init(void);
double simple_compute(double x);

int main(int argc, char **argv){
    int i;

    (void)argc;
    (void)argv;

    simple_init();

    for(i=0; i<10; i++){
        double sum;

        sum = simple_compute(0.87*i);
        printf("sum=%lf\n",sum);
    }

    return 0;
}

#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

extern inline unsigned long
get_um(void)
{
	unsigned long tmp;

	__asm__ __volatile__("mov %0=psr.um;;" :"=r"(tmp):: "memory" );

	return tmp;
}

int
main(int argc, char **argv)
{
	printf("umask=0x%lx up=%ld\n", get_um(), (get_um() >> 2) & 0x1);
	return 0;
}

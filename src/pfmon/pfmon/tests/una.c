#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __GNUC__
extern inline void
clear_psr_ac(void)
{
	__asm__ __volatile__("rum psr.ac;;" ::: "memory" );
}
#elif defined(__ECC) && defined(__INTEL_COMPILER)
#include <ia64intrin.h>
#define clear_psr_ac()	__rum(1<<3)
#else
#error "You need to define clear_psr_ac() for your compiler"
#endif


#define PFM_TEST_INVALID	-1
#define PFM_TEST_VALID		0

static union {
	unsigned long   l_tab[2];
	unsigned int    i_tab[4];
	unsigned short  s_tab[8];
	unsigned char   c_tab[16];
} __attribute__((__aligned__(32))) messy;

static int
do_una(int argc, char **argv)
{
	unsigned int *l, v;
	static int called;

	called++;
	l = (unsigned int *)(messy.c_tab+1);

	if (((unsigned long)l & 0x1) == 0) {
		printf("Data is not unaligned, can't run test\n");
		return  PFM_TEST_INVALID;
	}

	v = *l;
	v++;
	*l = v;

	if (v != called) return PFM_TEST_INVALID;


	return PFM_TEST_VALID;
}

int
main(int argc, char **argv)
{
	unsigned long count;
	int ret;

	/* let the hardware do the unaligned access */
	clear_psr_ac();

	count = argc > 1 ? strtoul(argv[1], NULL, 10) : 1;

	ret = PFM_TEST_VALID;

	while ( count-- && ret == PFM_TEST_VALID) ret = do_una(argc, argv);

	return ret;
}


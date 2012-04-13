/* $Id: ppc.h,v 1.2 2005/04/09 10:25:47 mikpe Exp $
 * PPC32-specific code for performance counters library.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */
#ifndef __LIB_PERFCTR_PPC_H
#define __LIB_PERFCTR_PPC_H

static __inline__ unsigned long get_tbl(void)
{
    unsigned long tbl;
    asm volatile("mftb %0" : "=r" (tbl));
    return tbl;
}
#define rdtscl(x)	do { (x) = get_tbl(); } while(0)

#define SPRN_PVR	0x11F
#define PVR_VER(pvr)	(((pvr) >> 16) & 0xFFFF)

#define SPRN_UPMC1	0x3A9
#define SPRN_UPMC2	0x3AA
#define SPRN_UPMC3	0x3AD
#define SPRN_UPMC4	0x3AE
#define SPRN_UPMC5	0x3A1
#define SPRN_UPMC6	0x3A2

#define __stringify_1(x)	#x
#define __stringify(x)		__stringify_1(x)
#define mfspr(rn)	({unsigned int rval; \
			asm volatile("mfspr %0," __stringify(rn) \
				: "=r" (rval)); rval; })

static __inline__ unsigned int read_pmc(unsigned int pmc)
{
    switch( pmc ) {
      default: /* impossible, but silences gcc warning */
      case 0:
	return mfspr(SPRN_UPMC1);
      case 1:
	return mfspr(SPRN_UPMC2);
      case 2:
	return mfspr(SPRN_UPMC3);
      case 3:
	return mfspr(SPRN_UPMC4);
      case 4:
	return mfspr(SPRN_UPMC5);
      case 5:
	return mfspr(SPRN_UPMC6);
    }
}
#define rdpmcl(pmc,x)	do { (x) = read_pmc((pmc)); } while(0)

#define vperfctr_has_rdpmc(vperfctr)	((vperfctr)->have_rdpmc)

extern void perfctr_info_cpu_init(struct perfctr_info*);

#endif /* __LIB_PERFCTR_PPC_H */

/* Maynard
 * PPC64-specific code for performance counters library.
 *
 */
#ifndef __LIB_PERFCTR_PPC64_H
#define __LIB_PERFCTR_PPC64_H

static __inline__ unsigned long get_tb(void)
{
	unsigned long tb;
	asm volatile("mftb %0" : "=r" (tb));
	return tb;
}

#define rdtscl(x)	do { (x) = (unsigned int) get_tb(); } while(0)

#define SPRN_UPMC1	0x303
#define SPRN_UPMC2	0x304
#define SPRN_UPMC3	0x305
#define SPRN_UPMC4	0x306
#define SPRN_UPMC5	0x307
#define SPRN_UPMC6	0x308
#define SPRN_UPMC7	0x309
#define SPRN_UPMC8	0x30a

#define MMCRA_SIHV	0x10000000UL /* state of MSR HV when SIAR set */
#define MMCRA_SIPR	0x08000000UL /* state of MSR PR when SIAR set */
#define MMCRA_SAMPLE_ENABLE 0x00000001UL /* enable sampling */

#define MMCR0_FC	0x80000000UL /* freeze counters. set to 1 on a perfmon exception */
#define MMCR0_FCS	0x40000000UL /* freeze in supervisor state */
#define MMCR0_KERNEL_DISABLE MMCR0_FCS
#define MMCR0_FCP	0x20000000UL /* freeze in problem state */
#define MMCR0_PROBLEM_DISABLE MMCR0_FCP
#define MMCR0_FCM1	0x10000000UL /* freeze counters while MSR mark = 1 */
#define MMCR0_FCM0	0x08000000UL /* freeze counters while MSR mark = 0 */
#define MMCR0_PMXE	0x04000000UL /* performance monitor exception enable */
#define MMCR0_FCECE	0x02000000UL /* freeze counters on enabled condition or event */
/* time base exception enable */
#define MMCR0_TBEE	0x00400000UL /* time base exception enable */
#define MMCR0_PMC1CE	0x00008000UL /* PMC1 count enable*/
#define MMCR0_PMCjCE	0x00004000UL /* PMCj count enable*/
#define MMCR0_TRIGGER	0x00002000UL /* TRIGGER enable */
#define MMCR0_PMAO	0x00000080UL /* performance monitor alert has occurred, set to 0 after handling exception */
#define MMCR0_SHRFC	0x00000040UL /* SHRre freeze conditions between threads */
#define MMCR0_FCTI	0x00000008UL /* freeze counters in tags inactive mode */
#define MMCR0_FCTA	0x00000004UL /* freeze counters in tags active mode */
#define MMCR0_FCWAIT	0x00000002UL /* freeze counter in WAIT state */
#define MMCR0_FCHV	0x00000001UL /* freeze conditions in hypervisor mode */

#define mfspr(rn)	({unsigned int rval; \
			asm volatile("mfspr %0,%1" : "=r"(rval) : "i"(rn)); \
			rval; })

static __inline__ unsigned int read_pmc(unsigned int pmc)
{
    switch (pmc) {
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
      case 6:
      	return mfspr(SPRN_UPMC7);
      case 7:
      	return mfspr(SPRN_UPMC8);
    }
}
#define rdpmcl(pmc,x)	do { (x) = read_pmc((pmc)); } while(0)

#define vperfctr_has_rdpmc(vperfctr)	((vperfctr)->have_rdpmc)

extern void perfctr_info_cpu_init(struct perfctr_info*);

#endif /* __LIB_PERFCTR_PPC64_H */

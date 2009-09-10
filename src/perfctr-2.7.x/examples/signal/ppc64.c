/* Maynard Johnson
 * PPC64-specific code.
 *
 */

#include <sys/ucontext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libperfctr.h"
#include "arch.h"
#include "ppc64.h"

unsigned long ucontext_pc(const struct ucontext *uc)
{
    /* glibc-2.3.3 (YDL4) changed the type of uc->uc_mcontext,
     * breaking code which worked in glibc-2.3.1 (YDL3.0.1).
     * This formulation works with both, and is cleaner than
     * selecting glibc-2.3.3 specific code with "#ifdef NGREG".
     */
    return uc->uc_mcontext.regs->nip;
}

void do_setup(const struct perfctr_info *info,
	      struct perfctr_cpu_control *cpu_control)
{
    memset(cpu_control, 0, sizeof *cpu_control);

    cpu_control->tsc_on = 1;
    cpu_control->nractrs = 0;
    cpu_control->nrictrs = 1;
    cpu_control->pmc_map[0] = 3;

    /* FLOPS COMPLETED */
    if ((info->cpu_type == PERFCTR_PPC64_POWER4) ||
        (info->cpu_type == PERFCTR_PPC64_POWER4p)) {
	cpu_control->ppc64.mmcr0 = 0x00000810ULL;
	cpu_control->ppc64.mmcr1 = 0x00000000420E84A0ULL;
	cpu_control->ppc64.mmcra = 0x00002000ULL;
    } else if (info->cpu_type == PERFCTR_PPC64_POWER5) {
        cpu_control->ppc64.mmcr0 = 0x00000000ULL;
        cpu_control->ppc64.mmcr1 = 0x0000000020202010ULL;
        cpu_control->ppc64.mmcra = 0x00000000ULL;
    } else if (info->cpu_type == PERFCTR_PPC64_970 ||
	       info->cpu_type == PERFCTR_PPC64_970MP) {
        cpu_control->ppc64.mmcr0 = 0x00000000ULL;
        cpu_control->ppc64.mmcr1 = 0x00000000001E0480ULL;
        cpu_control->ppc64.mmcra = 0x00002000ULL;
    }

    /* not kernel mode, enable PMCj interrupts */
    cpu_control->ppc64.mmcr0 |= MMCR0_FCS | MMCR0_PMCjCE;

    /* overflow after 100 events */
    cpu_control->ireset[0] = 0x80000000 - 100;


}

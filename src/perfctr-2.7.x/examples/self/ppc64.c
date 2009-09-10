/* Maynard Johnson
 * PPC64-specific code.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libperfctr.h"
#include "arch.h"

void do_setup(const struct perfctr_info *info,
	      struct perfctr_cpu_control *cpu_control)
{
    memset(cpu_control, 0, sizeof *cpu_control);
    cpu_control->tsc_on = 1;
    cpu_control->nractrs = 1;
    cpu_control->pmc_map[0] = 0;
/* Set up counter 1 to count PM_INST_CMPL.  Other counters are set up
 * to count some event, but are not used in this test.
 */
    if ((info->cpu_type == PERFCTR_PPC64_POWER4) || 
	(info->cpu_type == PERFCTR_PPC64_POWER4p)) {
	cpu_control->ppc64.mmcr0 = 0x4000090EULL;
	cpu_control->ppc64.mmcr1 = 0x1003400045F29420ULL;
	cpu_control->ppc64.mmcra = 0x00002000ULL;
    } else if (info->cpu_type == PERFCTR_PPC64_POWER5) {
	cpu_control->ppc64.mmcr0 = 0x00000000ULL;
	cpu_control->ppc64.mmcr1 = 0x8103000602CACE8EULL;
	cpu_control->ppc64.mmcra = 0x00000001ULL;
    } else if (info->cpu_type == PERFCTR_PPC64_970 ||
	       info->cpu_type == PERFCTR_PPC64_970MP) {
	cpu_control->ppc64.mmcr0 = 0x0000091EULL;
	cpu_control->ppc64.mmcr1 = 0x4003001005F09000ULL;
	cpu_control->ppc64.mmcra = 0x00002000ULL;
    }
}

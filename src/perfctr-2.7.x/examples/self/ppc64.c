/* Maynard
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
    if (info->cpu_type > PERFCTR_PPC64_GENERIC) {
	cpu_control->nractrs = 1;
	cpu_control->pmc_map[0] = 0;
	cpu_control->ppc64.mmcr0 = 0x4000090EULL;
	cpu_control->ppc64.mmcr1 = 0x1003400045F29420ULL;
	cpu_control->ppc64.mmcra = 0x00002000ULL;
    }
}

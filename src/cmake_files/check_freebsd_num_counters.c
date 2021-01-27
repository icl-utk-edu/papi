#include <unistd.h>
#include <pmc.h>

int main() {
    const struct pmc_cpuinfo *info;
    if (pmc_init() < 0) return 0;
    if (pmc_cpuinfo (&info) < 0) return 0;
    return info->pm_npmc-1;
}

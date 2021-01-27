#include <pmapi.h>

int main(int argc, char *argv[])
{
    int filter = PM_VERIFIED;
    pm_info2_t pminfo;
    pm_groups_info_t pmgroups;
    return pm_init ( filter,  &pminfo, &pmgroups, PM_CURRENT );
}

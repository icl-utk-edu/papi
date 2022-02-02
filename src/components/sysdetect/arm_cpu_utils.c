#include "sysdetect.h"
#include "arm_cpu_utils.h"
#include "os_cpu_utils.h"

int
arm_cpu_init( void )
{
    return CPU_SUCCESS;
}

int
arm_cpu_finalize( void )
{
    return CPU_SUCCESS;
}

int
arm_cpu_get_vendor( char *vendor )
{
    return os_cpu_get_vendor(vendor);
}

int
arm_cpu_get_name( char *name )
{
    return os_cpu_get_name(name);
}

int
arm_cpu_get_attribute( CPU_attr_e attr, int *value )
{
    return os_cpu_get_attribute(attr, value);
}

int
arm_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value )
{
    return os_cpu_get_attribute_at(attr, loc, value);
}

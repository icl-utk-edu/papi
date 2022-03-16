#include "sysdetect.h"
#include "arm_cpu_utils.h"
#include "os_cpu_utils.h"
#include <string.h>

#define VENDOR_ARM_ARM       65
#define VENDOR_ARM_BROADCOM  66
#define VENDOR_ARM_CAVIUM    67
#define VENDOR_ARM_FUJITSU   70
#define VENDOR_ARM_HISILICON 72
#define VENDOR_ARM_APM       80
#define VENDOR_ARM_QUALCOMM  81

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
    int papi_errno;

    char tmp[PAPI_MAX_STR_LEN];
    papi_errno = os_cpu_get_vendor(tmp);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int vendor_id;
    sscanf(tmp, "%x", &vendor_id);

    switch(vendor_id) {
        case VENDOR_ARM_ARM:
            strcpy(vendor, "Arm");
            break;
        case VENDOR_ARM_BROADCOM:
            strcpy(vendor, "Broadcom");
            break;
        case VENDOR_ARM_CAVIUM:
            strcpy(vendor, "Cavium");
            break;
        case VENDOR_ARM_FUJITSU:
            strcpy(vendor, "Fujitsu");
            break;
        case VENDOR_ARM_HISILICON:
            strcpy(vendor, "Hisilicon");
            break;
        case VENDOR_ARM_APM:
            strcpy(vendor, "Apm");
            break;
        case VENDOR_ARM_QUALCOMM:
            strcpy(vendor, "Qualcomm");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
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

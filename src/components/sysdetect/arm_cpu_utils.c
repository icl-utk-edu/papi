#include <stdio.h>
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
#define NAMEID_ARM_1176           0xb76
#define NAMEID_ARM_CORTEX_A7      0xc07
#define NAMEID_ARM_CORTEX_A8      0xc08
#define NAMEID_ARM_CORTEX_A9      0xc09
#define NAMEID_ARM_CORTEX_A15     0xc0f
#define NAMEID_ARM_CORTEX_A53     0xd03
#define NAMEID_ARM_CORTEX_A57     0xd07
#define NAMEID_ARM_NEOVERSE_N1    0xd0c
#define NAMEID_ARM_NEOVERSE_N2    0xd49
#define NAMEID_BROADCOM_THUNDERX2 0x516
#define NAMEID_CAVIUM_THUNDERX2   0x0af
#define NAMEID_FUJITSU_A64FX      0x001
#define NAMEID_HISILICON_KUNPENG  0xd01
#define NAMEID_APM_XGENE          0x000
#define NAMEID_QUALCOMM_KRAIT     0x040

static int name_id_arm_cpu_get_name( int name_id, char *name );
static int name_id_broadcom_cpu_get_name( int name_id, char *name );
static int name_id_cavium_cpu_get_name( int name_id, char *name );
static int name_id_fujitsu_cpu_get_name( int name_id, char *name );
static int name_id_hisilicon_cpu_get_name( int name_id, char *name );
static int name_id_apm_cpu_get_name( int name_id, char *name );
static int name_id_qualcomm_cpu_get_name( int name_id, char *name );

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
    int papi_errno;

    papi_errno = os_cpu_get_name(name);
    if (strlen(name) != 0) {
        return  papi_errno;
    }

    char tmp[PAPI_MAX_STR_LEN];
    papi_errno = os_cpu_get_vendor(tmp);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int vendor_id;
    sscanf(tmp, "%x", &vendor_id);

    int name_id;
    papi_errno = os_cpu_get_attribute(CPU_ATTR__CPUID_MODEL, &name_id);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    switch(vendor_id) {
        case VENDOR_ARM_ARM:
            papi_errno = name_id_arm_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_BROADCOM:
            papi_errno = name_id_broadcom_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_CAVIUM:
            papi_errno = name_id_cavium_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_FUJITSU:
            papi_errno = name_id_fujitsu_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_HISILICON:
            papi_errno = name_id_hisilicon_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_APM:
            papi_errno = name_id_apm_cpu_get_name(name_id, name);
            break;
        case VENDOR_ARM_QUALCOMM:
            papi_errno = name_id_qualcomm_cpu_get_name(name_id, name);
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
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

int
name_id_arm_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_ARM_1176:
            strcpy(name, "ARM1176");
            break;
        case NAMEID_ARM_CORTEX_A7:
            strcpy(name, "ARM Cortex A7");
            break;
        case NAMEID_ARM_CORTEX_A8:
            strcpy(name, "ARM Cortex A8");
            break;
        case NAMEID_ARM_CORTEX_A9:
            strcpy(name, "ARM Cortex A9");
            break;
        case NAMEID_ARM_CORTEX_A15:
            strcpy(name, "ARM Cortex A15");
            break;
        case NAMEID_ARM_CORTEX_A53:
            strcpy(name, "ARM Cortex A53");
            break;
        case NAMEID_ARM_CORTEX_A57:
            strcpy(name, "ARM Cortex A57");
            break;
        case NAMEID_ARM_NEOVERSE_N1:
            strcpy(name, "ARM Neoverse N1");
            break;
        case NAMEID_ARM_NEOVERSE_N2:
            strcpy(name, "ARM Neoverse N2");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_broadcom_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_BROADCOM_THUNDERX2:
            strcpy(name, "Broadcom ThunderX2");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_cavium_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_CAVIUM_THUNDERX2:
            strcpy(name, "Cavium ThunderX2");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_fujitsu_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_FUJITSU_A64FX:
            strcpy(name, "Fujitsu A64FX");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_hisilicon_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_HISILICON_KUNPENG:
            strcpy(name, "Hisilicon Kunpeng");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_apm_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_APM_XGENE:
            strcpy(name, "Applied Micro X-Gene");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

int
name_id_qualcomm_cpu_get_name( int name_id, char *name )
{
    int papi_errno = PAPI_OK;

    switch(name_id) {
        case NAMEID_QUALCOMM_KRAIT:
            strcpy(name, "ARM Qualcomm Krait");
            break;
        default:
            papi_errno = PAPI_ENOSUPP;
    }

    return papi_errno;
}

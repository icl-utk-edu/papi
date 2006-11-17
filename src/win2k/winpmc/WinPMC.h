/*****************************************************************************
******************************************************************************
*
*               WinPMC device driver header file
*
*               With thanks to Jan Houska of HUMUSOFT s.r.o.
*
******************************************************************************
*****************************************************************************/

#if     !defined(__WINPMC_H__)
#define __WINPMC_H__

#include "pmc_kernel.h"

// NT device naming
#define DEVICENAME            L"WinPMC"	  // name of the driver - used to open its handle

// File system device name.   When you execute a CreateFile call to open the
// device, use "\\.\WinPMC", or, given C's conversion of \\ to \, use
// "\\\\.\\WinPMC"

#define DEVICETYPE            ('PMC' | 0x8000)     // type of device - for non-standard devices use any number >=0x8000

//
// Define the IOCTL codes we will use.  The IOCTL code contains a command
// identifier, plus other information about the device, the type of access
// with which the file must have been opened, and the type of buffering.
//

//
// Device type           -- in the "User Defined" range."
//

#define PMC_TYPE 40000

#define HELLO \
    CTL_CODE( PMC_TYPE, 0x910, METHOD_BUFFERED, FILE_READ_ACCESS )

#define HELLONUM \
    CTL_CODE( PMC_TYPE, 0x911, METHOD_BUFFERED, FILE_READ_ACCESS )

#define TASKSWITCH \
    CTL_CODE( PMC_TYPE, 0x912, METHOD_BUFFERED, FILE_READ_ACCESS )

#define IOCTL_PMC_READ_TEST_STRING \
    CTL_CODE( PMC_TYPE, 0x920, METHOD_BUFFERED, FILE_READ_ACCESS )

#define IOCTL_PMC_VERSION_STRING \
    CTL_CODE( PMC_TYPE, 0x921, METHOD_BUFFERED, FILE_READ_ACCESS )

#define IOCTL_PMC_INFO \
    CTL_CODE( PMC_TYPE, 0x931, METHOD_BUFFERED, FILE_READ_ACCESS )

#define IOCTL_PMC_CONTROL \
    CTL_CODE( PMC_TYPE, 0x932, METHOD_BUFFERED, FILE_WRITE_ACCESS )

#endif // !defined(__WINPMC_H__)

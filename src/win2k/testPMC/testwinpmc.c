/*++

Module Name:

    testwinpmc.c

Abstract:  
	Assumes the WinPMC.sys driver is properly installed in the \system32\drivers directory;
	Exercises the driver by opening it, running some tests and displaying the results. 

Author:

    Dan Terpstra

Environment:

    Win32 console multi-threaded application

Revision History:

--*/

#include <windows.h>
#include <winioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <intrin.h>
#include "winpmc.h"

// put the driver through its paces to make sure it's there and active
static HANDLE LoadDriver(void)
{    
    HANDLE hDriver = INVALID_HANDLE_VALUE;

    // Try opening a static device driver. 
    hDriver = CreateFile("\\\\.\\WinPMC",
		     GENERIC_READ | GENERIC_WRITE, 
		     FILE_SHARE_READ | FILE_SHARE_WRITE,
		     0,                     // Default security
		     OPEN_EXISTING,
		     0,						// Don't Perform asynchronous I/O
		     0);                    // No template

    if (hDriver == INVALID_HANDLE_VALUE)
       printf("Bummer","Driver Load Failed: %d.\n",hDriver);
    return (hDriver);
}


// get the driver version string; also makes sure it's there and active
static void getDriverVersion(char *version, int size)
{    
    HANDLE hDriver;
    DWORD dwBytesReturned;
    BOOL  bReturnCode = FALSE;

    // Try opening a static device driver. 
    hDriver = LoadDriver();

    if (hDriver != INVALID_HANDLE_VALUE) {

	// Dispatch the PMC_VERSION_STRING IOCTL to our NT driver.
	bReturnCode = DeviceIoControl(hDriver,
				  IOCTL_PMC_VERSION_STRING,
				  NULL, 0, (int *)version, size,
				  &dwBytesReturned, NULL);
	CloseHandle(hDriver);
	version[dwBytesReturned-2] = 0; // make sure it's terminated
    } else version[0] = 0;
}


// put the driver through its paces to make sure it's there and active
static void DiagRDPMC(void)
{    
    HANDLE hDriver;
    unsigned __int64 i;

    // Try opening a static device driver. 
    hDriver = LoadDriver();
    if (hDriver != (INVALID_HANDLE_VALUE)) {
	printf("Ready to execute RDPMC from user space...\n");
    i = __readpmc(0);
    printf("We have successfully executed RDPMC from user space: 0x%x!\n", i);
	CloseHandle(hDriver);
    }
}


// put the driver through its paces to make sure it's there and active
static void HelloTest(void)
{    
    HANDLE hDriver;
    DWORD dwBytesReturned;
    BOOL  bReturnCode = FALSE;
    int iobuf[256];     // I/O buffer

    // Try opening a static device driver. 
    hDriver = LoadDriver();

    if (hDriver != INVALID_HANDLE_VALUE) {
	// Send a request to the driver. The request code is HELLO, no parameters
	bReturnCode = DeviceIoControl(hDriver, HELLO, NULL, 0, iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
	if (bReturnCode) {
		printf("HELLO RETURNED %d bytes: >%s<\n", dwBytesReturned, iobuf);
	}
	else 	printf("HELLO failed.\n");
	CloseHandle(hDriver);
    }
}


// put the driver through its paces to make sure it's there and active
static void HelloNumTest(void)
{    
    HANDLE hDriver;
    DWORD dwBytesReturned;
    BOOL  bReturnCode = FALSE;
    char szString[256]; // character buffer
    int iobuf[256];     // I/O buffer

    // Try opening a static device driver. 
    hDriver = LoadDriver();

    if (hDriver != INVALID_HANDLE_VALUE) {
	// Send a request to the driver. The request code is HELLONUM, one integer parameter
	iobuf[0] = 319;    // my favorite number :-)
	bReturnCode = DeviceIoControl(hDriver, HELLONUM, iobuf, sizeof(int), iobuf, sizeof(iobuf), &dwBytesReturned, NULL);
	if (bReturnCode) {
	  sprintf(szString, "HELLONUM RETURNED %d bytes: >%s<\n", dwBytesReturned, iobuf);
		printf("HELLONUM RETURNED %d bytes: >%s<\n", dwBytesReturned, iobuf);
	}
	else 	printf("HELLONUM failed.\n");
	CloseHandle(hDriver);
    }
}

VOID _cdecl main( ULONG argc, PCHAR argv[] )
{
  char text[512] = {"Fourscore and seven years ago..."};

  getDriverVersion(text, sizeof(text));
  printf(text);
  printf("\n");
  HelloTest();
  HelloNumTest();
  DiagRDPMC();
}
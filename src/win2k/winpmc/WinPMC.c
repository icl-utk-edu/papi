/*****************************************************************************
******************************************************************************
*
*           Device driver exec for Windows Performance Counter Access
*
*               With thanks to Jan Houska of HUMUSOFT s.r.o.
*
******************************************************************************
*****************************************************************************/


#include <ntddk.h>
#include <stdio.h>
#include "WinPMC.h"                           // contains I/O control codes

#define WELCOME_STRING		"Welcome to the Windows PMC Driver\n"
#define VERSION_STRING		"PMC Driver Version: 2006/08/11:2  \n"

/* variables */

static PDEVICE_OBJECT DeviceObject;		  // pointer to the device object


/*****************************************************************************
;*
;*		HELLOWORLD
;*              prints "Hello World"
;*
;*              Input:	buffer for the message
;*              Output:	length of output buffer
;*
;****************************************************************************/

static int HelloWorld(char* buf)
{

strcpy(buf, "Hello, world !!!");
return(strlen(buf)+1);
}



/*****************************************************************************
;*
;*		HELLOWORLDNUM
;*              prints "Hello World" with a count
;*
;*              Input:	buffer for the message
;*                	hello count
;*              Output:	length of output buffer
;*
;****************************************************************************/

static int HelloWorldNum(char* buf, int count)
{

sprintf(buf, "Hello, world %d times !!!", count);
return(strlen(buf)+1);
}



/*****************************************************************************
;*
;*		DISPATCH
;*		process the IRPs sent to this device
;*
;*              Input:	pointer to device object
;*			pointer to IRP
;*              Output:	status
;*
;****************************************************************************/

// The Dispatch routine processes all requests from applications. This branches
// according to the IRP packet requested.

static NTSTATUS Dispatch(PDEVICE_OBJECT DeviceObject, PIRP Irp)
{
   PULONG             ioBuffer;
   NTSTATUS           status;
   PIO_STACK_LOCATION irpStack;
   ULONG              ioControlCode;
   ULONG              inputBufferLength;
   ULONG              outputBufferLength;

   struct pmc_info info;


/* default is OK and nothing returned */

// Just to be safe we are OK if the code takes some unhandled path.

Irp->IoStatus.Status      = STATUS_SUCCESS;
Irp->IoStatus.Information = 0;

/* get a pointer to the current location in IRP */

// This gets the IRP which fully describes the request made by the application.

irpStack = IoGetCurrentIrpStackLocation(Irp);

/* branch according to the MajorFunction */

switch (irpStack->MajorFunction)
 {

// These two must be implemented and must succeed, but otherwise they may be empty.
// If you have something that should be done during CreateFile or CloseHandle, this
// is the place for it.

  case IRP_MJ_CREATE:
	status = kern_pmc_init();	// check cpu type & enable RDPMC
//	status = STATUS_SUCCESS;
  break;

 case IRP_MJ_CLOSE:
	kern_pmc_exit();			// turn off RDPMC before leaving
	status = STATUS_SUCCESS;
  break;

// This does all the actual work. We get the control code from the IRP and also the I/O buffer
// pointer and maximum length. The I/O buffer is used to transfer data - for large data packets
// there are other methods that avoid copying of the data but for our purposes I/O buffer is best.
// The length of data returned in the buffer must go to Irp->IoStatus.Information.

  case IRP_MJ_DEVICE_CONTROL:
    ioControlCode      = irpStack->Parameters.DeviceIoControl.IoControlCode;
    ioBuffer           = Irp->AssociatedIrp.SystemBuffer;
    inputBufferLength  = irpStack->Parameters.DeviceIoControl.InputBufferLength;
    outputBufferLength = irpStack->Parameters.DeviceIoControl.OutputBufferLength;

// The I/O control code specifies what to do. You are free to choose codes for individual actions but
// the lowest two bits are reserved and must be 0 if I/O buffer is to be used.

    switch (ioControlCode & ~0x3)
     {

	// Has no input parameters, returns "Hello world." string in the I/O buffer
   // For now this function has been coopted to display contents of the info structure.
   // It should ultimately be given its own entry point.
     case HELLO:
//        Irp->IoStatus.Information = HelloWorld((char *) ioBuffer);
//        status = STATUS_SUCCESS;
        status = kern_pmc_info(&info);
        if (status == STATUS_SUCCESS) {
            if (outputBufferLength > 90) // this is a guess on the required length
            {
               strcpy((char *)ioBuffer, "\nVendor  : ");
	            strncat((char *)ioBuffer, info.vendor, 12);
               sprintf(((char *)ioBuffer) + strlen((char *)ioBuffer), "\nFamily  : %d\nModel   : %d\nStepping: %d\nFeatures: 0x%8x", 
                  info.family, info.model, info.stepping, info.features);
	            Irp->IoStatus.Information = strlen((char *)ioBuffer)+1;
            }
            else status = STATUS_BUFFER_TOO_SMALL;
        }
      break;

	// Has one input parameters, returns "Hello world %d times." string in the I/O buffer
	// where %d is the value of the parameter.
      case HELLONUM:
        Irp->IoStatus.Information = HelloWorldNum((char *) ioBuffer, ioBuffer[0]);
        status = STATUS_SUCCESS;
      break;

	// Returns number of task switches since the driver was loaded.
   // This function is no longer relevant
      case TASKSWITCH:
        ioBuffer[0] = 0;
        Irp->IoStatus.Information = sizeof(int);
        status = STATUS_SUCCESS;
      break;

	// Returns a driver version string to the caller
	  case IOCTL_PMC_VERSION_STRING:
		if (outputBufferLength > strlen(VERSION_STRING)+1)
		{
			strcpy((char *)ioBuffer, VERSION_STRING);
			Irp->IoStatus.Information = strlen(VERSION_STRING)+1;
		    status = STATUS_SUCCESS;
		}
		else status = STATUS_BUFFER_TOO_SMALL;
	  break;

	// Returns a welcome string to the caller
	  case IOCTL_PMC_READ_TEST_STRING:
		if (outputBufferLength > strlen(WELCOME_STRING)+1)
		{
			strcpy((char *)ioBuffer, WELCOME_STRING);
			Irp->IoStatus.Information = strlen(WELCOME_STRING)+1;
		    status = STATUS_SUCCESS;
		}
		else status = STATUS_BUFFER_TOO_SMALL;
	  break;

	  case IOCTL_PMC_INFO:
		if (outputBufferLength > strlen(WELCOME_STRING)+1)
		{
		    status = kern_pmc_info(&info);
		    if (status == STATUS_SUCCESS) {
			*(struct pmc_info *)ioBuffer = info;
			Irp->IoStatus.Information = sizeof(struct pmc_info);
		    }
		}
		else status = STATUS_BUFFER_TOO_SMALL;
	  break;

	  case IOCTL_PMC_CONTROL:
		status = kern_pmc_control((struct pmc_control *)ioBuffer);
	        *(int *)ioBuffer = status;
		if (status >= 0) status = STATUS_SUCCESS;
		Irp->IoStatus.Information = sizeof(int);
	  break;

// Fails - invalid control code

      default:
        status = STATUS_INVALID_DEVICE_REQUEST;
     }
 }

// Complete the request after it's done. The IO_NO_INCREMENT means we don't want to bump up
// the caller's priority because there was no waiting for the request to complete.

IoCompleteRequest(Irp, IO_NO_INCREMENT);

return(status);
}


/*****************************************************************************
;*
;*		UNLOAD
;*		installable driver unload
;*
;*              Input:	pointer to driver object
;*              Output:	none
;*
;****************************************************************************/

// This function is called only during driver unload - that is, on explicit unload
// by e.g. "net stop xxxx". It is not called on system shutdown! Here we just undo some
// of the actions done during initialization.

static VOID Unload(PDRIVER_OBJECT DriverObject)
{
UNICODE_STRING deviceLinkUnicodeString;

// unhook the context switch

// KeSetSwapContextNotifyRoutine(NULL);

/* delete the symbolic link */

RtlInitUnicodeString(&deviceLinkUnicodeString, L"\\DosDevices\\"DEVICENAME);
IoDeleteSymbolicLink(&deviceLinkUnicodeString);

/* delete the device object */

IoDeleteDevice(DriverObject->DeviceObject);

}



/*****************************************************************************
;*
;*		DRIVERENTRY
;*		installable driver initialization entry point
;*
;*              Input:	pointer to driver object
;*			pointer to unicode string representing path to
;*                         driver-specific key in registry
;*              Output:	status
;*
;****************************************************************************/

// This function is called only during driver initialization. It creates the
// device object, establishes symbolic links, etc. You can also perform any
// one-time initialization tasks here.

NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath)
{
NTSTATUS           ntStatus;
UNICODE_STRING     deviceNameUnicodeString, deviceLinkUnicodeString;

/* create device object */

// This always is necessary to do. The device name must be unicode string, so we must create it
// first. You can allocate device extension here - this is nonpaged memory associated to the device where
// you can put your variables, but I prefer to use local variables and/or memory allocated by
// ExAllocatePool for this purpose, so I usually put 0 here. Device type should indicate if this is a mouse,
// printer, etc. - for a non-standard device like ours any number bigger than 0x8000 is good. The device
// characteristics doesn't apply for non-standard devices. If you want to access the device
// from multiple processes at the same time, you must set it to non-exclusive.

RtlInitUnicodeString(&deviceNameUnicodeString, L"\\Device\\"DEVICENAME);
ntStatus = IoCreateDevice (DriverObject,	     // pointer to driver object
                           0,			     // device extension size
                           &deviceNameUnicodeString, // device name
                           DEVICETYPE,		     // device type
                           0,			     // device characteristics
                           FALSE,                    // non-exclusive device
                           &DeviceObject);           // pointer to resulting device object
if (!NT_SUCCESS(ntStatus))
  return(ntStatus);

/* create dispatch points for device control and unload */

// This is also always necessary. Here you define the entry points to the driver. Normally you process all
// application requests in one routine (called Dispatch here). You must always process
// IRP_MJ_CREATE and IRP_MJ_CLOSE so that CreateFile and CloseHandle work from the application. Then you
// usually process IRP_MJ_READ, IRP_MJ_WRITE, IRP_MJ_DEVICE_CONTROL. Normally I use IRP_MJ_DEVICE_CONTROL for
// everything but sequential data streams coming from/to device - i.e data that look like they are being
// read from a file. We won't have any, so we don't process IRP_MJ_READ and IRP_MJ_WRITE.
// Then, you also need to register the Unload routine here.

DriverObject->MajorFunction[IRP_MJ_CREATE]         =
DriverObject->MajorFunction[IRP_MJ_CLOSE]          =
DriverObject->MajorFunction[IRP_MJ_DEVICE_CONTROL] = Dispatch;
DriverObject->DriverUnload                         = Unload;

/* create a symbolic link */

// This creates a symbolic DOS name. I'm not quite sure what is it for, but everyone creates it so let's do it as well.
// I believe it is used when you type "net start example" in the command prompt - the "example" is most probably
// this name. So, without doing this probably "net start xxxxxx" won't work, although I'm not 100% sure.

RtlInitUnicodeString(&deviceLinkUnicodeString, L"\\DosDevices\\"DEVICENAME);
ntStatus = IoCreateSymbolicLink(&deviceLinkUnicodeString, &deviceNameUnicodeString);
if (!NT_SUCCESS(ntStatus))
 {
  IoDeleteDevice(DeviceObject);   // delete device if link creation failed
  return(ntStatus);
 }

// That's all to initialization!

return(STATUS_SUCCESS);
}

// cpuinfo_papi.cpp : Defines the entry point for the console application.
// File by Kevin London
//


#include "cpuinfo.h"
#include <stdio.h>
#include <windows.h>


// Have to use the below function because sleep isn't reliable for determining
// Mhz on a laptop.....sigh.....--KSL
int mhz;
UINT_PTR mytimer;

void CALLBACK my_timer( UINT wTimerID, UINT msg, DWORD dwuser, DWORD dw1, DWORD dw2 );

void CALLBACK my_timer( UINT wTimerID, UINT msg, DWORD dwuser, DWORD dw1, DWORD dw2 ){
        static int count=0;
        static int mymhz=0;
        static ULARGE_INTEGER t1,t2;
        int tmpmhz=1;
        __asm {
                rdtsc
                mov t2.LowPart, eax
                mov t2.HighPart, edx
        }
        if ( count == 0 ){
                t1 = t2;
                count++;
                return;
        }
        else
                tmpmhz = (unsigned int)((t2.QuadPart-t1.QuadPart))/1000000;


        if ( tmpmhz > mymhz || mymhz == 0){
                t1 = t2;
                mymhz = tmpmhz;
                return;
        }
        else if ( count > 10 )
                mhz = mymhz;
        else {
	    if ( count>3&&mymhz<(tmpmhz+2)&&mymhz>(tmpmhz-2)) {
		mhz = mymhz;
		timeKillEvent( mytimer );
		return;
	    }
                t1 = t2;
                count++;
                return;
        }
        timeKillEvent( mytimer );
}

// The functions to initialize processor information,
// The only function a user should call is init_hwinfo -KSL

static int init_amd  ( struct wininfo * );
static int init_intel( struct wininfo * );

int init_hwinfo( struct wininfo * hwinfo) {
    volatile unsigned long val,val2, val3;
    SYSTEM_INFO sys_info;
    char vendor[13];
    int retval,dowork=0;



	// Initialize the entire array to 0
	memset(hwinfo, 0, sizeof(struct wininfo));

    GetSystemInfo(&sys_info); 

	hwinfo->arch = sys_info.wProcessorArchitecture;
	hwinfo->ncpu = sys_info.dwNumberOfProcessors;
	hwinfo->nnodes = 1;
	hwinfo->total_cpus = hwinfo->nnodes*hwinfo->ncpu;
	hwinfo->vendor = -1;

	if ( hwinfo->arch == PROCESSOR_ARCHITECTURE_INTEL );
	else 
		return -1;

// Lets see if CPUID exists, if not we are running on a PRE 486 processor	
 
	__asm {
		pushfd
		pop eax
		mov ebx, eax
		xor eax, 00200000h
		push eax
		popfd
		pushfd
		pop eax
		cmp eax, ebx
		jz NO_CPUID
		mov val, 1
		jmp END
	NO_CPUID:
		mov val, 0
	END:
	}
	if ( !val ) {
		return -1;
	}

	// Get the Vendor String
	__asm {
		mov EAX, 00h
		CPUID
		mov dword ptr[vendor], ebx
		mov dword ptr[vendor+4], edx
		mov dword ptr[vendor+8], ecx
	};


	vendor[12] = '\0';
	strcpy ( hwinfo->vendor_string, vendor );
	if ( !strcmp( vendor, "AuthenticAMD" ) )
		hwinfo->myvendor = AMD;
	else if ( !strcmp( vendor, "GenuineIntel") )
		hwinfo->myvendor = INTEL;
	else
		hwinfo->myvendor = UNKNOWN;


    // Get the standard information
	__asm {
		mov EAX, 01h
		CPUID
		mov val, eax
		mov val2, ebx
		mov val3, edx

	}

	hwinfo->family=((val&0xf00)>>8);
	hwinfo->processor_type=((val&0x3000)>>12);
	hwinfo->model=((val&0xf0)>>4);
	hwinfo->revision=(val&0xf);
	hwinfo->brand_id=(val2&0xff);

        mytimer = timeSetEvent( 1000, 0, my_timer, 0, TIME_PERIODIC );
        while ( !mhz ){
		if ( !(dowork%50000000) )
                	Sleep (1);
		else
			dowork++;
	}
        hwinfo->mhz = mhz;

	if ( hwinfo->myvendor == AMD )
		retval=init_amd( hwinfo );
	else if ( hwinfo->myvendor == INTEL )
		retval=init_intel( hwinfo );
	return retval;
}

static int init_amd( struct wininfo * hwinfo ) {
  // Setup Model Information
  hwinfo->nrctr = 4;
  if ( IS_AMDATHLON(hwinfo) )
	  sprintf ( hwinfo->model_string, "Athlon (%d)",hwinfo->model);
  else if ( IS_AMDDURON( hwinfo ) )
	  sprintf ( hwinfo->model_string, "Duron (%d)", hwinfo->model);
  else
	  return -1;
  return 1;
}

static int init_intel ( struct wininfo * hwinfo ) {
  volatile unsigned long val,val2, val3,val4,value;
  int count,i,j,tmp[4],k;

  // Setup cache information
  __asm {
	  mov EAX, 2
	  cpuid
	  mov val, EAX
	  mov val2, EBX
	  mov val3, ECX
	  mov val4, EDX
  }
  count = (val&0xff);
  for ( j=0; j<count;j++ ){
	  for ( i=0; i<4;i++ ){
		  if ( i == 0 )
			  value = val;
		  else if ( i==1 )
			  value = val2;
		  else if ( i==2 )
			  value = val3;
		  else
			  value = val4;
		  tmp[0] = (value&0xff);
		  tmp[1]= ((value&0xff00)>>8);
		  tmp[2]= ((value&0xff0000)>>16);
		  tmp[3]= ((value&0xff000000)>>24);
		  for ( k=0;k<4;k++ ){
			  if ( j==0 && i==0 && k==0 )
				  continue;
			  switch(tmp[k]){
			  default:
				  break;
			  case 0x41:
			  case 0x79:
			  case 0x81:
				  hwinfo->L2cache_size = 128;
				  break;
			  case 0x42:
			  case 0x7A:
			  case 0x82:
				  hwinfo->L2cache_size = 256;
				  break;
			  case 0x43:
			  case 0x7B:
			  case 0x83:
				  hwinfo->L2cache_size = 512;
				  break;
			  case 0x44:
			  case 0x7C:
			  case 0x84:
				  hwinfo->L2cache_size = 1024;
				  break;
			  case 0x45:
			  case 0x85:
				  hwinfo->L2cache_size = 2048;
				  break;
			  }
		  }
	  }
  }
  // Setup Model Information
  hwinfo->nrctr = 2;
  if ( IS_P4(hwinfo) ){
	  sprintf( hwinfo->model_string, "Pentium 4 (%d)",hwinfo->model);
	  hwinfo->nrctr = 18;
  }
  else if ( IS_P3XEON( hwinfo ) )
	  sprintf( hwinfo->model_string, "Pentium III Xeon (%d)",hwinfo->model);
  else if ( IS_P3(hwinfo) )
	  sprintf( hwinfo->model_string, "Pentium III (%d)",hwinfo->model);
  else if ( IS_P2(hwinfo) )
	  sprintf( hwinfo->model_string, "Pentium II (%d)", hwinfo->model);
  else if ( IS_CELERON(hwinfo) ){
	  if ( hwinfo->model == 6 || hwinfo->model == 5)
		sprintf( hwinfo->model_string, "Celeron Pentium II (%d)", hwinfo->model);
	  else if ( hwinfo->model == 8 )
		sprintf( hwinfo->model_string, "Celeron Pentium III (%d)", hwinfo->model);
	  else
		sprintf( hwinfo->model_string, "Celeron (%d)", hwinfo->model);
  }
  else{
	  sprintf(hwinfo->model_string, "Unknown" );
	  return -1;
  }
  return 1;
}

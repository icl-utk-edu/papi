/*
* This file determines the processor and memory about a processor for
* windows platforms.
* File: 	cpuinfo.c
* Author:	Kevin London
*		london@cs.utk.edu
* Mods:		<your name here>
*		<your email address>
*/

#include "cpuinfo.h"
#include <windows.h>
#include <Mmsystem.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>


void CALLBACK my_timer( UINT wTimerID, UINT msg, DWORD dwuser, DWORD dw1, DWORD dw2 );
static int init_amd ( struct wininfo *);
static int amd_proc( struct wininfo * hwinfo );
static int init_intel( struct wininfo * );
static int intel_proc( struct wininfo *hwinfo );

// Have to use the below function because sleep isn't reliable for determining
// Mhz on a laptop.....sigh.....
int mhz;
UINT_PTR mytimer;

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
		tmpmhz = (unsigned int)((t2.QuadPart-t1.QuadPart))/100000;
	

	if ( tmpmhz > mymhz || mymhz == 0){
		t1 = t2;
		mymhz = tmpmhz;
		return;
	}
	else if ( count > 10 )
		mhz = mymhz;
	else {
		if ( count>1 && mymhz < (tmpmhz+20) && mymhz > (tmpmhz-20) ){
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

int init_hwinfo( struct wininfo * hwinfo) {
    volatile unsigned long val,val2, val3;
    SYSTEM_INFO sys_info;
	char vendor[13];
    unsigned int dowork=1;
    MEMORYSTATUS stat;

	// Initialize features to does not exist
	memset(hwinfo, 0, sizeof(struct wininfo) );

    GetSystemInfo(&sys_info); 

	hwinfo->arch = sys_info.wProcessorArchitecture;
	hwinfo->ncpus = sys_info.dwNumberOfProcessors;
	hwinfo->pagesize = sys_info.dwPageSize;
	hwinfo->proc_level = sys_info.wProcessorLevel;
	hwinfo->nnodes = 1;
	hwinfo->total_cpus = hwinfo->nnodes*hwinfo->ncpus;

	if ( hwinfo->arch == PROCESSOR_ARCHITECTURE_INTEL );
	else if ( hwinfo->arch == PROCESSOR_ARCHITECTURE_ALPHA ){
		printf("We don't support the alpha windows version.\n");
		exit(1);
	}
	else if ( hwinfo->arch == PROCESSOR_ARCHITECTURE_PPC ) {
		printf("We don't support the PPC windows version.\n");
		exit(1);
	}
	else if ( hwinfo->arch == PROCESSOR_ARCHITECTURE_MIPS ) {
		printf("We don't support the MIPS windows version.\n");
		exit(1);
	}
	else {
		printf("We don't support this unknown processor.\n");
		exit(1);
    }
	// Lets see if CPUID exists, if not we are running on a PRE 486 processor	
    //		cli
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
		printf("CPUID not enabled, bailing out\n");
		exit(1);
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
		hwinfo->vendor = AMD;
	else if ( !strcmp( vendor, "GenuineIntel") )
		hwinfo->vendor = INTEL;
	else
		hwinfo->vendor = UNKNOWN;

    // Get the standard information
	__asm {
		mov EAX, 01h
		CPUID
		mov val, eax
		mov val2, ebx
		mov val3, edx

	}

	hwinfo->ext_family=((val&((1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)))>>20);
	hwinfo->ext_model=((val&((1<<19)|(1<<18)|(1<<17)|(1<<16)))>>16);
	hwinfo->family=((val&((1<<11)|(1<<10)|(1<<9)|(1<<8)))>>8);
	hwinfo->processor_type=((val&((1<<13)|(1<<12)))>>12);
	hwinfo->model=((val&((1<<7)|(1<<6)|(1<<5)|(1<<4)))>>4);
	hwinfo->stepping=(val&((1<<3)|(1<<2)|(1<<1)|(1<<0)));
	hwinfo->brand_id=(val2&((1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1)|(1<<0)));
	hwinfo->feature_flag = val3;
        hwinfo->revision=(val&0xf);
	
	mytimer = timeSetEvent( 100, 0, my_timer, 0, TIME_PERIODIC );

	while ( !mhz ){
		if ( !(dowork%10000000) ) {
			Sleep ( 1 );
		}
		else 
			dowork++;
	}

	hwinfo->mhz = mhz;
	// Setup Memory information
    GlobalMemoryStatus (&stat);
	hwinfo->total_phys = stat.dwTotalPhys;
	hwinfo->avail_phys = stat.dwAvailPhys;
	hwinfo->total_virt = stat.dwTotalVirtual;
	hwinfo->avail_virt = stat.dwAvailVirtual;
	if ( hwinfo->vendor == AMD )
		init_amd( hwinfo );
	else if ( hwinfo->vendor == INTEL )
		init_intel( hwinfo );
	return 1;
}

static int init_amd( struct wininfo * hwinfo ) {
  volatile unsigned long val,val2, val3,val4;
  
  hwinfo->processor_id = amd_proc( hwinfo );
  hwinfo->nrctr = 4;
  // Setup model information
  if ( IS_AMDOPTERON(hwinfo) )
	  strcpy ( hwinfo->model_string, "AMD K8 Opteron");
  else if ( IS_AMDATHLON(hwinfo) )
	  strcpy ( hwinfo->model_string, "AMD K7 Athlon");
  else if ( IS_AMDDURON( hwinfo ) )
	  strcpy ( hwinfo->model_string, "AMD K7 Duron" );
  else if ( IS_AMDK63(hwinfo) )
	  strcpy ( hwinfo->model_string, "K6-III");
  else if ( IS_AMDK62(hwinfo) )
	  strcpy ( hwinfo->model_string, "K6-2");
  else if ( IS_AMDK6(hwinfo) )
	  strcpy ( hwinfo->model_string, "K6");
  else if ( IS_AMDK5( hwinfo ) )
	  strcpy ( hwinfo->model_string, "K5");
  else if ( IS_AMD486( hwinfo ) )
	  strcpy ( hwinfo->model_string, "486 or 5x86");
  else
	  strcpy( hwinfo->model_string, "Unknown");

  // Setup features
  if ( hwinfo->feature_flag&(1<<31) )
	  hwinfo->DNOW = 1;
  if ( hwinfo->feature_flag&(1<<30) )
	  hwinfo->DNOW_EXT = 1;
  if ( hwinfo->feature_flag&(1<<24) )
	  hwinfo->FXSAVE = 1;
  if ( hwinfo->feature_flag&(1<<23) )
	  hwinfo->MMX = 1;
  if ( hwinfo->feature_flag&(1<<0) )
	  hwinfo->FPU = 1;
  if ( hwinfo->feature_flag&(1<<4) )
	  hwinfo->TSC = 1;
  if ( hwinfo->feature_flag&(1<<9) )
	  hwinfo->APIC = 1;
  if ( !IS_AMD486(hwinfo) ) {

  // Determine Cache charteristics
  if ( IS_AMDATHLON(hwinfo) || IS_AMDDURON(hwinfo) ) {
	  __asm {
		  mov EAX, 0x80000005
		  cpuid
          mov val, EAX
		  mov val2, EBX
		  mov val3, ECX
		  mov val4, EDX
	  }
  }
  else {
	  __asm {
		  mov EAX, 0x80000005
		  cpuid
		  mov val2, EBX
		  mov val3, ECX
		  mov val4, EDX
	  }

  }
  // NON-Athlon/Duron specific memory features
  hwinfo->L1datacache_assoc = ((val3&((1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18)|
						(1<<17)|(1<<16)))>>16);
  hwinfo->L1datacache_lines = ((val3&((1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|
						(1<<10)|(1<<9)|(1<<8)))>>8);
  hwinfo->L1datacache_linesize = (val3&((1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|
						(1<<1)|(1<<0)));
  hwinfo->L1datacache_size = ((val3&((1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|
						(1<<25)|(1<<24)))>>24);
  hwinfo->L1instcache_assoc = ((val4&((1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18)|
						(1<<17)|(1<<16)))>>16);
  hwinfo->L1instcache_lines = ((val4&((1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|
						(1<<10)|(1<<9)|(1<<8)))>>8);
  hwinfo->L1instcache_linesize = (val4&((1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|
						(1<<1)|(1<<0)));
  hwinfo->L1instcache_size = ((val4&((1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|
						(1<<25)|(1<<24)))>>24);
  if ( IS_AMDDURON(hwinfo) || IS_AMDATHLON(hwinfo) || IS_AMDK63(hwinfo) ) {
      __asm {
    	mov EAX, 0x80000006
		cpuid
		mov val, ECX
	  }
	  hwinfo->L2cache_assoc = ((val&((1<<15)|(1<<14)|(1<<13)|(1<<12)))>>12);
	  hwinfo->L2cache_lines = ((val&((1<<11)|(1<<10)|(1<<9)|(1<<8)))>>8);
	  hwinfo->L2cache_linesize = (val&(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|
							(1<<2)|(1<<1)|(1<<0));
	  hwinfo->L2cache_size = ((val&((1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|
					(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18)|
					(1<<17)|(1<<16)))>>16);
  }
  }
  return 1;
}

static int amd_proc( struct wininfo * hwinfo ) {
	unsigned int model = hwinfo->model;
	switch ( hwinfo->family ) {
	    default:
			return 0;
		case 4:
			return AMD_486;
		case 5:
			switch (model) {
				case 0:
				case 1:
				case 2:
				case 3:
					return AMD_K5;
				case 7:
				case 6:
					return AMD_K6;
				case 8:
					return AMD_K62;
				case 9:
					return AMD_K63;
				default:
					return PROC_UNKNOWN;
			}
		case 6:
			switch (model) {
				case 1:
				case 2:
				case 4:
					return AMD_ATHLON;
				case 3:
					return AMD_DURON;
				default:
					return PROC_UNKNOWN;
			}
		case 15:
			return AMD_OPTERON;
			/* There's a whole bunch of Opteron codes, but I'm not sure how
				much they matter. See: "Revision Guide for AMD Athlon 64 and
				AMD Opteron Processors" AMD Publication # 25759.pdf for details.
			switch (model) {
				case 5:
					return AMD_OPTERON;
				default:
					return PROC_UNKNOWN;
			}
			*/
	}
}

static int init_intel ( struct wininfo * hwinfo ) {
  volatile unsigned long val,val2, val3,val4,value;
  int count,i,j,tmp[4],k;
  char model[48];


  // Setup Features
  if ( hwinfo->feature_flag&(1<<0) )
	  hwinfo->FPU = 1;
  if ( hwinfo->feature_flag&(1<<4) )
	  hwinfo->TSC = 1;
  if ( hwinfo->feature_flag&(1<<9) )
	  hwinfo->APIC = 1;
  if ( hwinfo->feature_flag&(1<<18) )
	  hwinfo->SERIAL = 1;
  if ( hwinfo->feature_flag&(1<<22) )
	  hwinfo->ACPI = 1;
  if ( hwinfo->feature_flag&(1<<23) )
	  hwinfo->MMX = 1;
  if ( hwinfo->feature_flag&(1<<24) )
	  hwinfo->FXSAVE = 1;
  if ( hwinfo->feature_flag&(1<<25) )
	  hwinfo->SSE = 1;
  if ( hwinfo->feature_flag&(1<<26) )
	  hwinfo->SSE2 = 1;
  if ( hwinfo->feature_flag&(1<<29) )
	  hwinfo->TM = 1;

  // Setup cache information
  __asm {
	  mov EAX, 2
	  cpuid
	  mov val, EAX
	  mov val2, EBX
	  mov val3, ECX
	  mov val4, EDX
  }
  count = (val&((1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1)|(1<<0)));
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
		  tmp[0] = (value&((1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1)|(1<<0)));
		  tmp[1]= ((value&((1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)))>>8);
		  tmp[2]= ((value&((1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18)|(1<<17)|(1<<16)))>>16);
		  tmp[3]= ((value&((1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)))>>24);
		  for ( k=0;k<4;k++ ){
			  if ( j==0 && i==0 && k==0 )
				  continue;
			  switch(tmp[k]){
			  default:
				  break;
			  case 0x06:
				  hwinfo->L1instcache_size = 8;
				  hwinfo->L1instcache_assoc = 4;
				  hwinfo->L1instcache_linesize = 32;
				  break;
			  case 0x08:
				  hwinfo->L1instcache_size = 16;
				  hwinfo->L1instcache_assoc = 4;
				  hwinfo->L1instcache_linesize = 32;
				  break;
			  case 0x0A:
				  hwinfo->L1datacache_size = 8;
				  hwinfo->L1datacache_assoc = 2;
				  hwinfo->L1datacache_linesize = 32;
				  break;
			  case 0x0C:
				  hwinfo->L1datacache_size = 16;
				  hwinfo->L1datacache_assoc = 4;
				  hwinfo->L1datacache_linesize = 32;
				  break;
			  case 0x22:
				  hwinfo->L3cache_assoc = 4;
				  hwinfo->L3cache_linesize = 64;
				  hwinfo->L3cache_size = 512;
				  break;
			  case 0x23:
				  hwinfo->L3cache_assoc = 8;
				  hwinfo->L3cache_linesize = 64;
				  hwinfo->L3cache_size = 1024;
				  break;
			  case 0x25:
				  hwinfo->L3cache_assoc = 8;
				  hwinfo->L3cache_linesize = 64;
				  hwinfo->L3cache_size = 2048;
				  break;
			  case 0x29:
				  hwinfo->L3cache_assoc = 8;
				  hwinfo->L3cache_linesize = 64;
				  hwinfo->L3cache_size = 4096;
				  break;
			  case 0x40:
				  if ( IS_P4(hwinfo ) )
					  hwinfo->L3cache_size = 0;
				  else if ( hwinfo->family == 6 )
					  hwinfo->L2cache_size = 0;
				  break;
			  case 0x41:
				  hwinfo->L2cache_size = 128;
				  hwinfo->L2cache_assoc = 4;
				  hwinfo->L2cache_linesize = 32;
				  break;
			  case 0x42:
				  hwinfo->L2cache_size = 256;
				  hwinfo->L2cache_assoc = 4;
				  hwinfo->L2cache_linesize = 32;
				  break;
			  case 0x43:
				  hwinfo->L2cache_size = 512;
				  hwinfo->L2cache_assoc = 4;
				  hwinfo->L2cache_linesize = 32;
				  break;
			  case 0x44:
				  hwinfo->L2cache_size = 1024;
				  hwinfo->L2cache_assoc = 4;
				  hwinfo->L2cache_linesize = 32;
				  break;
			  case 0x45:
				  hwinfo->L2cache_size = 2048;
				  hwinfo->L2cache_assoc = 4;
				  hwinfo->L2cache_linesize = 32;
				  break;
			  case 0x66:
				  hwinfo->L1datacache_assoc = 4;
				  hwinfo->L1datacache_linesize = 64;
				  hwinfo->L1datacache_size = 8;
				  hwinfo->L1cache_sectored = 1;
				  break;
			  case 0x67:
				  hwinfo->L1datacache_assoc = 4;
				  hwinfo->L1datacache_linesize = 64;
				  hwinfo->L1datacache_size = 16;
				  hwinfo->L1cache_sectored = 1;
				  break;
			  case 0x68:
				  hwinfo->L1datacache_assoc = 4;
				  hwinfo->L1datacache_linesize = 64;
				  hwinfo->L1datacache_size = 32;
				  hwinfo->L1cache_sectored = 1;
				  break;
			  case 0x70:
				  hwinfo->L1tracecache_assoc = 8;
				  hwinfo->L1tracecache_size = 12;
				  break;
			  case 0x71:
				  hwinfo->L1tracecache_assoc = 8;
				  hwinfo->L1tracecache_size = 16;
				  break;
			  case 0x72:
				  hwinfo->L1tracecache_assoc = 8;
				  hwinfo->L1tracecache_size = 32;
				  break;
			  case 0x79:
				  hwinfo->L2cache_assoc = 8;
				  hwinfo->L2cache_linesize = 64;
				  hwinfo->L2cache_size = 128;
				  hwinfo->L2cache_codedata = 1;
				  hwinfo->L2cache_sectored = 1;
				  break;
			  case 0x7A:
				  hwinfo->L2cache_assoc = 8;
				  hwinfo->L2cache_linesize = 64;
				  hwinfo->L2cache_size = 256;
				  hwinfo->L2cache_codedata = 1;
				  hwinfo->L2cache_sectored = 1;
				  break;
			  case 0x7B:
				  hwinfo->L2cache_assoc = 8;
				  hwinfo->L2cache_linesize = 64;
				  hwinfo->L2cache_size = 512;
				  hwinfo->L2cache_codedata = 1;
				  hwinfo->L2cache_sectored = 1;
				  break;
			  case 0x7C:
				  hwinfo->L2cache_assoc = 8;
				  hwinfo->L2cache_linesize = 64;
				  hwinfo->L2cache_size = 1024;
				  hwinfo->L2cache_codedata = 1;
				  hwinfo->L2cache_sectored = 1;
				  break;
			  case 0x81:
				  hwinfo->L2cache_assoc = 8;
				  hwinfo->L2cache_linesize = 32;
				  hwinfo->L2cache_size = 128;
				  hwinfo->L2cache_codedata = 1;
			  case 0x82:
				  hwinfo->L2cache_assoc = 8;
				  hwinfo->L2cache_linesize = 32;
				  hwinfo->L2cache_size = 256;
				  hwinfo->L2cache_codedata = 1;
				  break;
			  case 0x83:
				  hwinfo->L2cache_assoc = 8;
				  hwinfo->L2cache_linesize = 32;
				  hwinfo->L2cache_size = 512;
				  hwinfo->L2cache_codedata = 1;
				  break;
			  case 0x84:
				  hwinfo->L2cache_assoc = 8;
				  hwinfo->L2cache_linesize = 32;
				  hwinfo->L2cache_size = 1024;
				  hwinfo->L2cache_codedata = 1;
				  break;
			  case 0x85:
				  hwinfo->L2cache_assoc = 8;
				  hwinfo->L2cache_linesize = 32;
				  hwinfo->L2cache_size = 2048;
				  hwinfo->L2cache_codedata = 1;
				  break;
			  }
		  }
	  }
  }
  hwinfo->processor_id = intel_proc( hwinfo );
  if ( IS_P4(hwinfo) )
	  hwinfo->nrctr = 18;
  else
  	hwinfo->nrctr = 2;
  // Setup Model String
  __asm{
	  mov EAX, 80000000h
	  CPUID
	  mov val, EAX
  };
  i = (int)(val-0x80000000);
  if ( i > 1 ) {
   __asm {
		mov EAX, 80000002h
		CPUID
		mov dword ptr[model], eax
		mov dword ptr[model+4], ebx
		mov dword ptr[model+8], ecx
		mov dword ptr[model+12], edx
	};
   if ( i > 2 ) {
	   __asm{
		mov EAX, 80000003h
		CPUID
		mov dword ptr[model+16], eax
		mov dword ptr[model+20], ebx
		mov dword ptr[model+24], ecx
		mov dword ptr[model+28], edx
	};
   }
   if ( i > 3 ) {
	   __asm{
		mov EAX, 80000004h
		CPUID
		mov dword ptr[model+32], eax
		mov dword ptr[model+36], ebx
		mov dword ptr[model+40], ecx
		mov dword ptr[model+44], edx
	};
   }
    model[47] = '\0';
    i = 0;
    while (model[i++] < 'A'); /* strip leading format chars */
    strcpy( hwinfo->model_string, &model[i-1]);
  }
  else {
    if ( IS_P4(hwinfo) )
	  strcpy( hwinfo->model_string, "Pentium 4");
    else if ( IS_P3(hwinfo) )
	  strcpy( hwinfo->model_string, "Pentium III");
    else if ( IS_P2(hwinfo) )
	  strcpy( hwinfo->model_string, "Pentium II");
    else if ( IS_MOBILE(hwinfo) )
	  strcpy( hwinfo->model_string, "Pentium Mobile");
    else if ( IS_CELERON(hwinfo) ){
	  if ( hwinfo->model == 6 || hwinfo->model == 5)
		strcpy( hwinfo->model_string, "Celeron Pentium II");
	  else if ( hwinfo->model == 8 )
		strcpy( hwinfo->model_string, "Celeron Pentium III");
	  else
		strcpy( hwinfo->model_string, "Celeron");
	}
    else if (IS_PPRO(hwinfo) )
	  strcpy( hwinfo->model_string, "Pentium Pro");
    else if (IS_PENTIUM(hwinfo) )
	  strcpy( hwinfo->model_string, "Pentium");
    else if ( IS_P3_XEON( hwinfo ) )
	  strcpy( hwinfo->model_string, "Pentium III Xeon");
	else if ( IS_486(hwinfo) ){
		if ( hwinfo->model==8 )
			strcpy( hwinfo->model_string, "DX4");
		else if ( hwinfo->model==7 )
			strcpy( hwinfo->model_string, "Write-Back enhanced DX2");
		else if ( hwinfo->model==5 )
			strcpy( hwinfo->model_string, "SX2");
		else if ( hwinfo->model==4 )
			strcpy( hwinfo->model_string, "486SL");
		else if ( hwinfo->model==3 )
			strcpy( hwinfo->model_string, "DX2");
		else if ( hwinfo->model==2 )
			strcpy( hwinfo->model_string, "486SX");
		else
			strcpy( hwinfo->model_string, "486DX");
	}
    else if ( IS_OVERDRIVE(hwinfo) ){
		if ( hwinfo->family == 6 )
		   strcpy( hwinfo->model_string, "Pentium II Overdrive");
		else 
	       strcpy( hwinfo->model_string, "Pentium Overdrive");
	}
    else
	  strcpy( hwinfo->model_string, "Unknown");
  }
  return 1;
}

static int intel_proc( struct wininfo * hwinfo ) {
	unsigned int model = hwinfo->model;
	switch ( hwinfo->brand_id ) {
		case 0x1:
			return INTEL_CELERON;
		case 0x2:
			return INTEL_P3;
		case 0x3:
			return INTEL_P3_XEON;
		case 0x4:
			return INTEL_P3;
		case 0x6:
			return INTEL_P3;
		case 0x7:
			return INTEL_CELERON;
		case 0x8:
			return INTEL_P4;
		case 0x9:
			return INTEL_P4;
		case 0xA:
			return INTEL_CELERON;
		case 0xB:
			return INTEL_XEON;
		case 0xC:
			return INTEL_XEON;
		case 0xE:
			return INTEL_P4;
		case 0xF:
			return INTEL_CELERON;
		case 0x13:
			return INTEL_CELERON;
		case 0x16:
			return INTEL_MOBILE;
		default:
			break;
	}
	if ( hwinfo->processor_type == 1 ) return INTEL_OVERDRIVE;
	else{
		switch ( hwinfo->family ){
		  case 6:
			  if ( model==11 ) return INTEL_P3;
			  else if ( model==7 ) { // Should be no celeron so no need to check if cache=0
				  if ( hwinfo->L2cache_size <=512 ) return INTEL_P3;
				  else return INTEL_P3_XEON;
			  }
			  else if ( model==6 ) return INTEL_CELERON;
			  else if ( model==5 ) {
				  if ( hwinfo->L2cache_size==0 ) return INTEL_CELERON;
				  else if ( hwinfo->L2cache_size>=1024 ) return INTEL_P2_XEON;
				  else return INTEL_P2;
			  }
			  else if ( model==3 ) return INTEL_P2;
			  else if ( model==1 ) return INTEL_PPRO;
			  else if ( model==8 ) break;  // This will be caught by brand_id so no need todo it again
			  else if ( model==10 ) return INTEL_P3_XEON; // Put last because brand_id should get this
			  break;
		  case 5:
			  if ( model==4 || model==2 || model==1) return INTEL_PENTIUM;
			  break;
		  case 4:
			  return INTEL_486;
		  case 15: // This should be taken care of brand_id
			  if ( model==0 || model==1 ) return INTEL_P4;
			  break;
		  default:
			break;
		}
	}
	return PROC_UNKNOWN;
}


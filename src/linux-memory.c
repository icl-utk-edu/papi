/* 
* File:    linux-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#ifdef __LINUX__
#include <limits.h>
#endif
#include SUBSTRATE
#include <stdio.h>
int init_amd( PAPI_mem_info_t * mem_info );
int init_intel( PAPI_mem_info_t * mem_info );

int get_memory_info( PAPI_mem_info_t * mem_info, int cpu_type ){
   int retval = 0;

/*
   if ( !check_cpuid() ) {
	return PAPI_ESBSTR;
   }
*/

   /* Defaults to Intel which is *probably* a safe assumption -KSL */
   switch ( cpu_type ) {
	case PERFCTR_X86_AMD_K7:
		retval = init_amd(mem_info);
		break;
	default:
		retval = init_intel(mem_info);
		break;
   }
   DBG((stderr,"Detected L1: %d L2: %d  L3: %d\n",
	mem_info->total_L1_size, mem_info->L2_cache_size, 
	mem_info->L3_cache_size));
   return retval;
}

int init_amd( PAPI_mem_info_t * mem_info ) {
   volatile unsigned long tmp,tmp2,tmp3,tmp4;

   DBG((stderr,"Initializing AMD memory\n"));
   /* AMD level 1 cache info */
   __asm__ __volatile__ ("movl $0x80000005, %%eax;"
	"cpuid;"
	"movl %%eax, %0;"
	"movl %%ebx, %1;"
	"movl %%ecx, %2;"
	"movl %%edx, %3;"
	:"=r"(tmp),"=r"(tmp2),"=r"(tmp3),"=r"(tmp4)
	:
	:"%eax","%ebx");
   mem_info->itlb_size      = ((tmp&0xff)*2048);
   mem_info->itlb_assoc     = ((tmp&0xff00)>>8);
   mem_info->dtlb_size      = (((tmp&0xff0000)>>16)*2048);
   mem_info->dtlb_assoc     = ((tmp&0xff000000)>>24);
   if ( !mem_info->itlb_size ) {  /* 4k pages */
       mem_info->itlb_size      = ((tmp2&0xff)*4);
       mem_info->itlb_assoc     = ((tmp2&0xff00)>>8);
   }
   if ( !mem_info->dtlb_size ) { /* 4k pages */
       mem_info->dtlb_size      = (((tmp2&0xff0000)>>16)*4);
       mem_info->dtlb_assoc     = ((tmp2&0xff000000)>>24);
   }
   mem_info->L1_dcache_size = ((tmp3&0xff000000)>>24);
   mem_info->L1_dcache_assoc= ((tmp3&0xff0000)>>16);
   mem_info->L1_dcache_lines= ((tmp3&0xff00)>>8);
   mem_info->L1_dcache_linesize=((tmp3&0xff));
   mem_info->L1_icache_size = ((tmp4&0xff000000)>>24);
   mem_info->L1_icache_assoc= ((tmp4&0xff0000)>>16);
   mem_info->L1_icache_lines= ((tmp4&0xff00)>>8);
   mem_info->L1_icache_linesize=((tmp4&0xff));
   mem_info->total_L1_size=mem_info->L1_icache_size+mem_info->L1_dcache_size;

   /* AMD level 2 cache info */
   __asm__ __volatile__ ("movl $0x80000006, %%eax;"
	"cpuid;"
	"movl %%ecx, %0;"
	:"=r"(tmp)
	:
	: "%eax", "%ebx", "%ecx", "%edx" );	
   mem_info->L2_cache_size  = ((tmp&0xffff0000)>>16);
   mem_info->L2_cache_assoc = ((tmp&0xf000)>>12);
   mem_info->L2_cache_lines = ((tmp&0xf00)>>8);
   mem_info->L2_cache_linesize=((tmp&0xff));

   /* AMD doesn't have Level 3 cache yet..... */
   return PAPI_OK;
}

int init_intel( PAPI_mem_info_t * mem_info ) {
   volatile unsigned long tmp,tmp2,tmp3,tmp4,value;
   int i,j,k,count;

  DBG((stderr,"Initializing Intel Memory\n"));
  /* All of Intels cache info is in 1 call to cpuid
   * however it is a table lookup :(
   */
   __asm__("movl $0x02, %%eax;"
	"cpuid;"
	"movl %%eax, %0;"
	"movl %%ebx, %1;"
	"movl %%ecx, %2;"
	"movl %%edx, %3;"
	: "=r"(tmp), "=r"(tmp2),"=r"(tmp3),"=r"(tmp4)
	:
	: "%eax", "%edx" );

   count = (0xff&tmp);
   for ( j=0; j<count; j++ ) {
     for ( i=0;i<4;i++){
	if(i==0) value = tmp;
	else if (i==1) value = tmp2;
	else if (i==2) value = tmp3;
	else value = tmp4;
	for (k=0;k<=4;k++){
		if(i==0&&j==0&&k==0) {
			value=value>>8;
			continue;
		}
		switch((value&0xff)){
			  case 0x01:
				  mem_info->itlb_size = 128;
				  mem_info->itlb_assoc= 4;
				  break;
			  case 0x02:
				  mem_info->itlb_size = 8;
				  mem_info->itlb_assoc= 1;
				  break;
			  case 0x03:
				  mem_info->dtlb_size = 256;
				  mem_info->dtlb_assoc= 4;
				  break;
			  case 0x04:
				  mem_info->dtlb_size = 32;
				  mem_info->dtlb_assoc= 4;
				  break;
                          case 0x06:
                                  mem_info->L1_icache_size = 8;
                                  mem_info->L1_icache_assoc = 4;
                                  mem_info->L1_icache_linesize = 32;
                                  break;
                          case 0x08:
                          case 0x15:
                                  mem_info->L1_icache_size = 16;
                                  mem_info->L1_icache_assoc = 4;
                                  mem_info->L1_icache_linesize = 32;
                                  break;
                          case 0x0A:
                                  mem_info->L1_dcache_size = 8;
                                  mem_info->L1_dcache_assoc = 2;
                                  mem_info->L1_dcache_linesize = 32;
                                  break;
                          case 0x0C:
                          case 0x10:
                                  mem_info->L1_dcache_size = 16;
                                  mem_info->L1_dcache_assoc = 4;
                                  mem_info->L1_dcache_linesize = 32;
                                  break;
                          case 0x1A:
                                  mem_info->L2_cache_size = 96;
                                  mem_info->L2_cache_assoc = 6;
                                  mem_info->L2_cache_linesize = 64;
                                  break;
                          case 0x22:
                                  mem_info->L3_cache_assoc = 4;
                                  mem_info->L3_cache_linesize = 64;
                                  mem_info->L3_cache_size = 512;
                                  break;
                          case 0x23:
                                  mem_info->L3_cache_assoc = 8;
                                  mem_info->L3_cache_linesize = 64;
                                  mem_info->L3_cache_size = 1024;
                                  break;
                          case 0x25:
                                  mem_info->L3_cache_assoc = 8;
                                  mem_info->L3_cache_linesize = 64;
                                  mem_info->L3_cache_size = 2048;
                                  break;
                          case 0x29:
                                  mem_info->L3_cache_assoc = 8;
                                  mem_info->L3_cache_linesize = 64;
                                  mem_info->L3_cache_size = 4096;
                                  break;
                          case 0x39:
                                  mem_info->L2_cache_assoc = 4;
                                  mem_info->L2_cache_linesize = 64;
                                  mem_info->L2_cache_size = 128;
                                  break;
                          case 0x3C:
                                  mem_info->L2_cache_assoc = 4;
                                  mem_info->L2_cache_linesize = 64;
                                  mem_info->L2_cache_size = 256;
                                  break;
                          case 0x40:
/* Need to fix this 
                                  if ( IS_P4(mem_info ) )
                                          mem_info->L3_cache_size = 0;
                                  else if ( mem_info->family == 6 )
*/
                                          mem_info->L2_cache_size = 0;
                                  break;
                          case 0x41:
                                  mem_info->L2_cache_size = 128;
                                  mem_info->L2_cache_assoc = 4;
                                  mem_info->L2_cache_linesize = 32;
                                  break;
                          case 0x42:
                                  mem_info->L2_cache_size = 256;
                                  mem_info->L2_cache_assoc = 4;
                                  mem_info->L2_cache_linesize = 32;
                                  break;
                          case 0x43:
                                  mem_info->L2_cache_size = 512;
                                  mem_info->L2_cache_assoc = 4;
                                  mem_info->L2_cache_linesize = 32;
                                  break;
                          case 0x44:
                                  mem_info->L2_cache_size = 1024;
                                  mem_info->L2_cache_assoc = 4;
                                  mem_info->L2_cache_linesize = 32;
                                  break;
                          case 0x45:
                                  mem_info->L2_cache_size = 2048;
                                  mem_info->L2_cache_assoc = 4;
                                  mem_info->L2_cache_linesize = 32;
                                  break;
			  case 0x50:
			  case 0x51:
			  case 0x52:
			  case 0x5B:
			  case 0x5C:
			  case 0x5D:
				  /*There is no way to determine
		 		   * the size since the page size
				   * can be 4K,2M or 4M and there
				   * is no way to determine it
				   * Sigh -KSL
				   */
				  mem_info->itlb_size=-1;
				  mem_info->itlb_assoc=1;
				  break;
                          case 0x66:
                                  mem_info->L1_dcache_assoc = 4;
                                  mem_info->L1_dcache_linesize = 64;
                                  mem_info->L1_dcache_size = 8;
                                  break;
                          case 0x67:
                                  mem_info->L1_dcache_assoc = 4;
                                  mem_info->L1_dcache_linesize = 64;
                                  mem_info->L1_dcache_size = 16;
                                  break;
                          case 0x68:
                                  mem_info->L1_dcache_assoc = 4;
                                  mem_info->L1_dcache_linesize = 64;
                                  mem_info->L1_dcache_size = 32;
                                  break;
                          case 0x70:
                                  mem_info->L1_icache_assoc = 8;
                                  mem_info->L1_icache_size = 12;
                                  break;
                          case 0x71:
                                  mem_info->L1_icache_assoc = 8;
                                  mem_info->L1_icache_size = 16;
                                  break;
                          case 0x72:
                                  mem_info->L1_icache_assoc = 8;
                                  mem_info->L1_icache_size = 32;
                                  break;
                          case 0x77:
                                  mem_info->L1_icache_size = 16;
                                  mem_info->L1_icache_assoc = 4;
                                  mem_info->L1_icache_linesize = 64;
                                  break;
                          case 0x79:
                                  mem_info->L2_cache_assoc = 8;
                                  mem_info->L2_cache_linesize = 64;
                                  mem_info->L2_cache_size = 128;
                                  break;
                          case 0x7A:
                                  mem_info->L2_cache_assoc = 8;
                                  mem_info->L2_cache_linesize = 64;
                                  mem_info->L2_cache_size = 256;
                                  break;
                          case 0x7B:
                                  mem_info->L2_cache_assoc = 8;
                                  mem_info->L2_cache_linesize = 64;
                                  mem_info->L2_cache_size = 512;
                                  break;
                          case 0x7C:
                                  mem_info->L2_cache_assoc = 8;
                                  mem_info->L2_cache_linesize = 64;
                                  mem_info->L2_cache_size = 1024;
                                  break;
                          case 0x7E:
                                  mem_info->L2_cache_assoc = 8;
                                  mem_info->L2_cache_linesize = 128;
                                  mem_info->L2_cache_size = 256;
                                  break;
                          case 0x81:
                                  mem_info->L2_cache_assoc = 8;
                                  mem_info->L2_cache_linesize = 32;
                                  mem_info->L2_cache_size = 128;
                          case 0x82:
                                  mem_info->L2_cache_assoc = 8;
                                  mem_info->L2_cache_linesize = 32;
                                  mem_info->L2_cache_size = 256;
                                  break;
                          case 0x83:
                                  mem_info->L2_cache_assoc = 8;
                                  mem_info->L2_cache_linesize = 32;
                                  mem_info->L2_cache_size = 512;
                                  break;
                          case 0x84:
                                  mem_info->L2_cache_assoc = 8;
                                  mem_info->L2_cache_linesize = 32;
                                  mem_info->L2_cache_size = 1024;
                                  break;
                          case 0x85:
                                  mem_info->L2_cache_assoc = 8;
                                  mem_info->L2_cache_linesize = 32;
                                  mem_info->L2_cache_size = 2048;
                                  break;
                          case 0x88:
                                  mem_info->L3_cache_assoc = 4;
                                  mem_info->L3_cache_linesize = 64;
                                  mem_info->L3_cache_size = 2048;
                                  break;
                          case 0x89:
                                  mem_info->L3_cache_assoc = 4;
                                  mem_info->L3_cache_linesize = 64;
                                  mem_info->L3_cache_size = 4096;
                                  break;
                          case 0x8A:
                                  mem_info->L3_cache_assoc = 4;
                                  mem_info->L3_cache_linesize = 64;
                                  mem_info->L3_cache_size = 8192;
                                  break;
                          case 0x8D:
                                  mem_info->L3_cache_assoc = 12;
                                  mem_info->L3_cache_linesize = 128;
                                  mem_info->L3_cache_size = 3096;
                                  break;
		}
		value=value>>8;
	   }
	}
    }
  mem_info->total_L1_size = mem_info->L1_icache_size+mem_info->L1_dcache_size;
  return PAPI_OK;
}

/* Checks to see if cpuid exists on this processor, if
 * it doesn't it is pre pentium K6 series that we don't
 * support.
 */

int  check_cpuid(){
volatile unsigned long val;
   __asm__ __volatile__("pushfl;"
                "pop %%eax;"
                "movl %%eax, %%ebx;"
                "xor $0x00200000,%%eax;"
                "push %%eax;"
                "popfl;"
                "pop %%eax;"
                "cmp %%eax, %%ebx;"
                "jz NO_CPUID;"
                "movl $1, %0;"
                "jmp END;"
        "NO_CPUID:"
                "movl $0, %0;"
        "END:"
        :"=r"(val));
	return (int) val;
}

long _papi_hwd_get_dmem_info(int option){
   pid_t pid = getpid();
   char pfile[256];
   FILE * fd;
   int tmp;
   unsigned int vsize,rss;

   sprintf(pfile, "/proc/%d/stat", pid);
   if((fd=fopen(pfile,"r")) == NULL ) {
        DBG((stderr,"PAPI_get_dmem_info can't open /proc/%d/stat\n",pid));
        return(PAPI_ESYS);
   }
  fgets(pfile, 256, fd);
  fclose(fd);
  
   /* Scan through the information */
  sscanf(pfile,"%d %s %c %d %d %d %d %d %u %u %u %u %u %d %d %d %d %d %d %d %d %d %u %u", 
	&tmp,pfile,pfile,&tmp,&tmp,&tmp,&tmp,&tmp,
	&tmp,&tmp,&tmp,&tmp, &tmp,&tmp,&tmp,&tmp,
	&tmp, &tmp,&tmp,&tmp,&tmp,&tmp, &vsize,&rss );
/*
	&tmp[0],pfile,pfile,&tmp[1],&tmp[2],&tmp[3],&tmp[4],&tmp[5],
	&tmp[6],&tmp[7],&tmp[8],&tmp[9], &tmp[10],&tmp[11],&tmp[12],&tmp[13],
	&tmp[14], &tmp[15],&tmp[16],&tmp[17],&tmp[18],&tmp[19], &vsize,&rss );
*/
 switch(option){
   case PAPI_GET_RESSIZE:
        return(rss);
   case PAPI_GET_SIZE:
 	tmp=getpagesize();
 	if ( tmp == 0 ) tmp = 1;
        return((vsize/tmp));
   default:
        return(PAPI_EINVAL);
  }
}

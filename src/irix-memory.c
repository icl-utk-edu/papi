/*
* File:    irix-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
* Author:  Zhou Min
*	   min@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include <invent.h>
#include <sys/systeminfo.h>
#include SUBSTRATE


#define TLB_R4  96*32
#define TLB_R5  96*32
#define TLB_R8  384*32
#define TLB_R10  128*32
#define TLB_R12  128*32

inventory_t *getinvent (void);
int get_memory_info( PAPI_hw_info_t * mem_info ){
	inventory_t *curr;
	long count;
	int chiptype;
	char ptype[80];

	count = 80;
	while ( (curr=getinvent())!= NULL ) {
		if ((curr->inv_class == INV_MEMORY ) && (curr->inv_type==INV_DCACHE)) {	
			mem_info->L1_dcache_size = curr->inv_state /1024;
		}
		if ((curr->inv_class == INV_MEMORY ) && (curr->inv_type==INV_ICACHE)) {	
			mem_info->L1_icache_size = curr->inv_state /1024;
		}
		if ((curr->inv_class == INV_MEMORY ) && (curr->inv_type==INV_SIDCACHE))	
			mem_info->L2_cache_size = curr->inv_state / 1024 ;
	}
	mem_info->L1_size = mem_info->L1_dcache_size + mem_info->L1_icache_size;

	mem_info->L1_dcache_linesize = 32;
	mem_info->L1_dcache_lines = mem_info->L1_dcache_size / mem_info->L1_dcache_linesize;
	mem_info->L1_dcache_assoc = 2;
	mem_info->L1_icache_linesize = 64;
	mem_info->L1_icache_lines = mem_info->L1_icache_size / mem_info->L1_icache_linesize;
	mem_info->L1_icache_assoc = 2;

	mem_info->L2_cache_assoc =2;
	mem_info->L2_cache_linesize = 128;
	mem_info->L2_cache_lines = mem_info->L2_cache_size / mem_info->L2_cache_linesize;

	sysinfo(_MIPS_SI_PROCESSORS, ptype, count);
	sscanf(ptype+1,"%d", &chiptype);
	switch (chiptype) {
		case 4000:
			mem_info->L1_tlb_size = TLB_R4/1024;
			break;
		case 5000:
			mem_info->L1_tlb_size = TLB_R5/1024;
			break;
		case 8000:
			mem_info->L1_tlb_size = TLB_R8/1024;
			break;
		case 10000:
			mem_info->L1_tlb_size = TLB_R10/1024;
			break;
		case 12000:
			mem_info->L1_tlb_size = TLB_R12/1024;
			break;
		default:
            break;
/*
			mem_info->total_tlb_size = TLB_R4/1024;
*/
	}
    return PAPI_OK;
}

long _papi_hwd_get_dmem_info(int option){
   pid_t pid = getpid();
   prpsinfo_t info;
   char pfile[256];
   int fd;

   sprintf(pfile, "/proc/%05d", (int)pid);
   if((fd=open(pfile,O_RDONLY)) <0 ) {
        DBG((stderr,"PAPI_get_dmem_info can't open /proc/%d\n",(int)pid));
        return(PAPI_ESYS);
   }
   if(ioctl(fd, PIOCPSINFO,  &info)<0){
        return(PAPI_ESYS);
   }
   close(fd);
 switch(option){
   case PAPI_GET_RESSIZE:
        return(info.pr_rssize);
   case PAPI_GET_SIZE:                                      
        return(info.pr_size);
   default:
        return(PAPI_EINVAL);
  }
}

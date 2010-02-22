/*
* File:    solaris-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    Philip J. Mucci
*          mucci@cs.utk.edu
*/

#include "papi.h"
#include "papi_internal.h"

int
get_memory_info( PAPI_hw_info_t * mem_info )
{
	return PAPI_OK;
}

int
_papi_hwd_get_dmem_info( PAPI_dmem_info_t * d )
{
	/* This function has been reimplemented 
	   to conform to current interface.
	   It has not been tested.
	   Nor has it been confirmed for completeness.
	   dkt 05-10-06
	 */

	FILE *fd;
	struct psinfo psi;

	if ( ( fd = fopen( "/proc/self/psinfo", "r" ) ) == NULL ) {
		SUBDBG( "fopen(/proc/self) errno %d", errno );
		return ( PAPI_ESYS );
	}

	fread( ( void * ) &psi, sizeof ( struct psinfo ), 1, fd );
	fclose( fd );

	d->pagesize = getpagesize(  );
	d->size = ( ( 1024 * psi.pr_rssize ) / d->pagesize );
	d->resident = ( ( 1024 * psi.pr_size ) / d->pagesize );
	d->high_water_mark = PAPI_EINVAL;
	d->shared = PAPI_EINVAL;
	d->text = PAPI_EINVAL;
	d->library = PAPI_EINVAL;
	d->heap = PAPI_EINVAL;
	d->locked = PAPI_EINVAL;
	d->stack = PAPI_EINVAL;

	return ( PAPI_OK );

/*  Depending on OS we may need this, so going to leave
 *  the code here for now. -KSL
   pid_t pid = getpid();
   psinfo_t info;
   char pfile[256];
   long pgsz=getpagesize();
   int fd;

   sprintf(pfile, "/proc/%05d", pid);
   if((fd=open(pfile,O_RDONLY)) <0 ) {
        SUBDBG((stderr,"PAPI_get_dmem_info can't open /proc/%d\n",pid));
        return(PAPI_ESYS);
   }
   if(ioctl(fd, PIOCPSINFO,  &info)<0){
        return(PAPI_ESYS);
   }
   close(fd);
 switch(option){
   case PAPI_GET_RESSIZE:
        return(((1024*info.pr_rssize)/pgsz));
   case PAPI_GET_SIZE:
        return(((1024*info.pr_size)/pgsz));
   default:
        return(PAPI_EINVAL);
  }
  */
}


#include "papi.h"
#include "papi_internal.h"

#if defined(sun)&&defined(sparc)

int _papi_hwd_update_shlib_info(void)
{
   char fname[80], name[PATH_MAX];
   prmap_t newp;
   int count, t_index;
   FILE * map_f;
   void * vaddr;
   Dl_info dlip;
   PAPI_address_map_t *tmp = NULL;

   sprintf(fname, "/proc/%d/map", getpid());
   map_f = fopen(fname, "r");

   /* count the entries we need */
   count =0;
   t_index=0;
   while ( fread(&newp, sizeof(prmap_t), 1, map_f) > 0 ) {
      vaddr = (void*)(1+(newp.pr_vaddr)); // map base address 
      if (dladdr(vaddr, &dlip) > 0) {
         count++;
         if ((newp.pr_mflags & MA_EXEC) && (newp.pr_mflags & MA_READ) ) {
            if ( !(newp.pr_mflags & MA_WRITE)) 
               t_index++;
         }
         strcpy(name,dlip.dli_fname);
         if (strcmp(_papi_hwi_system_info.exe_info.address_info.name, 
                          basename(name))== 0 ) {
            if ((newp.pr_mflags & MA_EXEC) && (newp.pr_mflags & MA_READ) ) {
               if ( !(newp.pr_mflags & MA_WRITE)) {
                  _papi_hwi_system_info.exe_info.address_info.text_start = 
                                      (caddr_t) newp.pr_vaddr;
                  _papi_hwi_system_info.exe_info.address_info.text_end =
                                      (caddr_t) (newp.pr_vaddr+newp.pr_size);
               } else {
                  _papi_hwi_system_info.exe_info.address_info.data_start = 
                                      (caddr_t) newp.pr_vaddr;
                  _papi_hwi_system_info.exe_info.address_info.data_end =
                                      (caddr_t) (newp.pr_vaddr+newp.pr_size);
               }  
            }
         }
      } 

   }
   rewind(map_f);
   tmp = (PAPI_address_map_t *) calloc(t_index-1, sizeof(PAPI_address_map_t));

   if (tmp == NULL)
      error_return(PAPI_ENOMEM, "Error allocating shared library address map");
   t_index=-1;
   while ( fread(&newp, sizeof(prmap_t), 1, map_f) > 0 ) {
      vaddr = (void*)(1+(newp.pr_vaddr)); // map base address
      if (dladdr(vaddr, &dlip) > 0) {  // valid name
         strcpy(name,dlip.dli_fname);
         if (strcmp(_papi_hwi_system_info.exe_info.address_info.name, 
                          basename(name))== 0 ) 
            continue;
         if ((newp.pr_mflags & MA_EXEC) && (newp.pr_mflags & MA_READ) ) {
            if ( !(newp.pr_mflags & MA_WRITE)) {
               t_index++;
               tmp[t_index].text_start = (caddr_t) newp.pr_vaddr;
               tmp[t_index].text_end =(caddr_t) (newp.pr_vaddr+newp.pr_size);
               strncpy(tmp[t_index].name, dlip.dli_fname, PAPI_MAX_STR_LEN);
            } else {
               tmp[t_index].data_start = (caddr_t) newp.pr_vaddr;
               tmp[t_index].data_end = (caddr_t) (newp.pr_vaddr+newp.pr_size);
            }
         }
      }
   }

   fclose(map_f);

   if (_papi_hwi_system_info.shlib_info.map)
         free(_papi_hwi_system_info.shlib_info.map);
   _papi_hwi_system_info.shlib_info.map = tmp;
   _papi_hwi_system_info.shlib_info.count = t_index+1;

   return(PAPI_OK);
}

#elif (defined(mips) && defined(sgi)) 

#include <sys/syscall.h>
#include <dlfcn.h>
/*
typedef struct DL_INFO {
     const char * dli_fname;
     void       * dli_fbase;
     const char * dli_sname;
     void       * dli_saddr;
     int          dli_version;
     int          dli_reserved1;
     long         dli_reserved[4];
} Dl_info;
*/
void * dladdr(void *address, Dl_info *dl)
{
   return( _rld_new_interface(_RLD_DLADDR,address,dl));
}

const char * getbasename(const char *fname)
{
    const char *temp;

    temp = strrchr(fname, '/');
    if( temp == NULL) {temp=fname; return temp;}
       else return temp+1;
}

int _papi_hwd_update_shlib_info(void)
{
   char procfile[100];
   prmap_t *p;
   Dl_info dlip;
   void * vaddr;
   int i, nmaps, err, fd, nmaps_allocd, count, t_index;
   PAPI_address_map_t *tmp = NULL;

   /* Construct the name of our own "/proc/${PID}" file, then open it. */
   sprintf(procfile, "/proc/%d", getpid());
   fd = open(procfile, O_RDONLY);
   if (fd < 0)
      return(PAPI_ESYS);
   /* Find out (approximately) how many map entries we have. */
   err = ioctl(fd, PIOCNMAP, &nmaps);
   if (err < 0) {
      return(PAPI_ESYS);
   }

   /* create space to hold that many entries, plus a generous buffer,
    * since PIOCNMAP can lie.
    */
   nmaps_allocd = 2 * nmaps + 10;
   p = (prmap_t *) calloc(nmaps_allocd, sizeof(prmap_t));
   if (p == NULL)
      return(PAPI_ENOMEM);
   err = ioctl(fd, PIOCMAP, p);
   if (err < 0) {
      return(PAPI_ESYS);
   }

   /* Basic cross-check between PIOCNMAP & PIOCMAP. Complicated by the
     * fact that PIOCNMAP *always* seems to report one less than PIOCMAP,
     * so we quietly ignore that little detail...

       The PIOCMAP entry on the proc man page says that
       one more is needed, so a minimum  one more than
       is returned by PIOCNMAP is required.
    */
   for (i = 0; p[i].pr_size != 0 && i < nmaps_allocd; ++i)
   ; /*empty*/
   if (i!= nmaps){ 
      printf(" i=%d nmaps=%d \n", i, nmaps);
   }

   count=0;
   t_index=0;
   for (i = 0; p[i].pr_size != 0 && i < nmaps_allocd; ++i)
   {
      vaddr =  (void *)(1+p[i].pr_vaddr); /* map base address */
      if (dladdr(vaddr, &dlip) > 0 ) 
      {
         count++;
         /* count text segments */
         if ((p[i].pr_mflags & MA_EXEC) && (p[i].pr_mflags & MA_READ) ) {
            if ( !(p[i].pr_mflags & MA_WRITE))
               t_index++;
         }
         if (strcmp(_papi_hwi_system_info.exe_info.address_info.name, 
                          getbasename(dlip.dli_fname))== 0 ) 
         {
            if ( (p[i].pr_mflags & MA_EXEC))
            {
                _papi_hwi_system_info.exe_info.address_info.text_start = 
                                   (caddr_t) p[i].pr_vaddr;
                _papi_hwi_system_info.exe_info.address_info.text_end =
                                   (caddr_t) (p[i].pr_vaddr+p[i].pr_size);
            } else {
                _papi_hwi_system_info.exe_info.address_info.data_start = 
                                   (caddr_t) p[i].pr_vaddr;
                _papi_hwi_system_info.exe_info.address_info.data_end =
                                   (caddr_t) (p[i].pr_vaddr+p[i].pr_size);
            }
         }

      };
   }
   tmp = (PAPI_address_map_t *) calloc(t_index-1, sizeof(PAPI_address_map_t));
   if (tmp == NULL)
      return(PAPI_ENOMEM);
   t_index=-1;

   /* assume the records about the same shared object are saved in the
      array contiguously. This may not be right, but it seems work fine.
    */
   for (i = 0; p[i].pr_size != 0 && i < nmaps_allocd; ++i)
   {
      vaddr =  (void *)(1+p[i].pr_vaddr); /* map base address */
      if (dladdr(vaddr, &dlip) > 0 ) 
      {
         if (strcmp(_papi_hwi_system_info.exe_info.address_info.name, 
                          getbasename(dlip.dli_fname))== 0 ) 
            continue;
         if ( (p[i].pr_mflags & MA_EXEC)) {
            t_index++;
            tmp[t_index].text_start = (caddr_t) p[i].pr_vaddr;
            tmp[t_index].text_end =(caddr_t) (p[i].pr_vaddr+p[i].pr_size);
            strncpy(tmp[t_index].name, dlip.dli_fname, PAPI_MAX_STR_LEN);
         } else {
            tmp[t_index].data_start = (caddr_t) p[i].pr_vaddr;
            tmp[t_index].data_end = (caddr_t) (p[i].pr_vaddr+p[i].pr_size);
         }
      }
   }
   if (_papi_hwi_system_info.shlib_info.map)
      free(_papi_hwi_system_info.shlib_info.map);
   _papi_hwi_system_info.shlib_info.map = tmp;
   _papi_hwi_system_info.shlib_info.count = t_index+1;

   return(PAPI_OK);
}
#endif 

/*
* File:    linux-ia64-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    <your name here>
*          <your email address>
*/

#include "papi.h"
#include SUBSTRATE
#include "papi_preset.h"
#include "papi_internal.h"
#include "papi_protos.h"

inline void get_cpu_info(unsigned int *rev, unsigned int *model, unsigned int *family, unsigned int *archrev);
void fline ( FILE *fp, char *buf );
int get_number(char *buf);

/* 
   Note that by convention, DATA information is stored in array index 1,
   while INST and UNIFIED information is stored in array index 0.
   Also levels 1, 2 and 3 are stored in array index 0, 1, 2.
   Thus (clevel - 1) produces the right level index.
*/

int _papi_hwd_get_memory_info(PAPI_hw_info_t * mem_info, int cpu_type)
{
   unsigned int rev,model,family,archrev;
   int retval = 0;
   FILE *f;
   int clevel = 0, cindex = -1;
   char buf[1024];
   int num, i, j;
   PAPI_mh_level_t *L = mem_info->mem_hierarchy.level;

   f = fopen("/proc/pal/cpu0/cache_info","r");

   if (!f)
      error_return(PAPI_ESYS, "fopen(/proc/pal/cpu0/cache_info returned < 0");

   while (!feof(f)) {
      fline(f, buf);
      if ( buf[0] == '\0' ) break;
      if (  !strncmp(buf, "Data Cache", 10) ) {
         cindex = 1;
         clevel = get_number( buf );
         L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_DATA;
      }
      else if ( !strncmp(buf, "Instruction Cache", 17) ) {
         cindex = 0;
         clevel = get_number( buf );
         L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_INST;
      }
      else if ( !strncmp(buf, "Data/Instruction Cache", 22)) {
         cindex = 0;
         clevel = get_number( buf );
         L[clevel - 1].cache[cindex].type = PAPI_MH_TYPE_UNIFIED;
      }
      else {
         if ( (clevel == 0 || clevel > 3) && cindex >= 0)
            error_return(PAPI_EBUG, "Cache type could not be recognized, send /proc/pal/cpu0/cache_info");

         if ( !strncmp(buf, "Size", 4) ) {
            num = get_number( buf );
            L[clevel - 1].cache[cindex].size = num;
         }
         else if ( !strncmp(buf, "Associativity", 13) ) {
            num = get_number( buf );
            L[clevel - 1].cache[cindex].associativity = num;
         }
         else if ( !strncmp(buf, "Line size", 9) ) {
            num = get_number( buf );
            L[clevel - 1].cache[cindex].line_size = num;
            L[clevel - 1].cache[cindex].num_lines = L[clevel - 1].cache[cindex].size/num;
         }
      }
   } 

   fclose(f);

   f = fopen("/proc/pal/cpu0/vm_info","r");
   /* No errors on fopen as I am not sure this is always on the systems */
   if ( f != NULL ) {
      cindex = -1;
      clevel = 0;
      while (!feof(f)) {
         fline(f, buf);
         if ( buf[0] == '\0' ) break;
         if (  !strncmp(buf, "Data Translation", 16) ) {
            cindex = 1;
	         clevel = get_number( buf );
            L[clevel - 1].tlb[cindex].type = PAPI_MH_TYPE_DATA;
         }
         else if ( !strncmp(buf, "Instruction Translation", 23) ){
            cindex = 0;
            clevel = get_number( buf );
            L[clevel - 1].tlb[cindex].type = PAPI_MH_TYPE_INST;
         }
         else {
	         if ( (clevel == 0 || clevel > 2) && cindex >= 0)
	            error_return(PAPI_EBUG, "TLB type could not be recognized, send /proc/pal/cpu0/vm_info");

	         if ( !strncmp(buf, "Number of entries", 17) ){
	            num = get_number( buf );
               L[clevel - 1].tlb[cindex].num_entries = num;
	         }
  	         else if ( !strncmp(buf, "Associativity", 13) ) {
	            num = get_number( buf );
               L[clevel - 1].tlb[cindex].associativity = num;
	         }
         }
      } 
      fclose(f);
   }

   /* Compute and store the number of levels of hierarchy actually used */
   for (i=0; i<PAPI_MAX_MEM_HIERARCHY_LEVELS; i++) {
      for (j=0; j<2; j++) {
         if (L[i].tlb[j].type != PAPI_MH_TYPE_EMPTY ||
            L[i].cache[j].type != PAPI_MH_TYPE_EMPTY)
            mem_info->mem_hierarchy.levels = i+1;
      }
   }

   get_cpu_info(&rev,&model,&family,&archrev);
   return retval;
}

int get_number( char *buf ){
   char numbers[] = "0123456789";
   int num;
   char *tmp, *end;

   tmp = strpbrk(buf, numbers);
   if ( tmp != NULL ){
	end = tmp;
	while(isdigit(*end)) end++;
	*end='\0';
        num = atoi(tmp);
        return(num);
    }
    else {
        error_return(PAPI_EBUG, "Cache type could not be recognized, send /proc/pal/cpu0/cache_info");
    }
  return(-1);
}

long _papi_hwd_get_dmem_info(int option)
{
   char pfile[256];
   FILE *fd;
   int tmp;
   unsigned int vsize, rss;

   if ((fd = fopen("/proc/self/stat", "r")) == NULL) {
      DBG((stderr, "PAPI_get_dmem_info can't open /proc/self/stat\n"));
      return (PAPI_ESYS);
   }
   fgets(pfile, 256, fd);
   fclose(fd);

   /* Scan through the information */
   sscanf(pfile,
          "%d %s %c %d %d %d %d %d %u %u %u %u %u %d %d %d %d %d %d %d %d %d %u %u", &tmp,
          pfile, pfile, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp,
          &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &vsize, &rss);
   switch (option) {
   case PAPI_GET_RESSIZE:
      return (rss);
   case PAPI_GET_SIZE:
      tmp = getpagesize();
      if (tmp == 0)
         tmp = 1;
      return ((vsize / tmp));
   default:
      return (PAPI_EINVAL);
   }
}


void fline ( FILE *fp, char *rline ) {
  char *tmp,*end,c;

  tmp = rline;
  end = &rline[1023];
   
  memset(rline, '\0', 1024);

  do {
    if ( feof(fp))  return;
    c = getc(fp);
  } while (isspace(c) || c == '\n' || c == '\r');

  ungetc( c, fp);

  for(;;) {
    if ( feof(fp) ) {
       return;
    }
    c = getc( fp);
    if ( c == '\n' || c == '\r' )
      break;
    *tmp++ = c;
    if ( tmp == end ) {
       *tmp = '\0';
       return;
    }
  }
  return;
} 
 
inline void get_cpu_info(unsigned int *rev, unsigned int *model, unsigned int *family, unsigned int *archrev)
{
        unsigned long r;

        asm ("mov %0=cpuid[%r1]" : "=r"(r) : "rO"(3));
        *rev = (r>>8)&0xff;
        *model = (r>>16)&0xff;
        *family = (r>>24)&0xff;
        *archrev = (r>>32)&0xff;
}


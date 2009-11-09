/*
* File:    linux-ia64-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
*
* Mods:    <Brian Sheely>
*          <bsheely@eecs.utk.edu>
*/

#include "papi.h"
#include "papi_internal.h"

void fline ( FILE *fp, char *buf );
int get_number(char *buf);

/* 
   Note that by convention, DATA information is stored in array index 1,
   while INST and UNIFIED information is stored in array index 0.
   Also levels 1, 2 and 3 are stored in array index 0, 1, 2.
   Thus (clevel - 1) produces the right level index.
*/

int _ia64_get_memory_info(PAPI_hw_info_t * mem_info, int cpu_type)
{
   int retval = 0;
   FILE *f;
   int clevel = 0, cindex = -1;
   char buf[1024];
   int num, i, j;
   PAPI_mh_level_t *L = mem_info->mem_hierarchy.level;

   f = fopen("/proc/pal/cpu0/cache_info","r");

   if (!f)
     { PAPIERROR("fopen(/proc/pal/cpu0/cache_info returned < 0"); return(PAPI_ESYS); }

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
	   { PAPIERROR("Cache type could not be recognized, please send /proc/pal/cpu0/cache_info"); return(PAPI_EBUG); }

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
		   { PAPIERROR("TLB type could not be recognized, send /proc/pal/cpu0/vm_info"); return(PAPI_EBUG); }

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
   for (i=0; i<PAPI_MH_MAX_LEVELS; i++) {
      for (j=0; j<2; j++) {
         if (L[i].tlb[j].type != PAPI_MH_TYPE_EMPTY ||
            L[i].cache[j].type != PAPI_MH_TYPE_EMPTY)
            mem_info->mem_hierarchy.levels = i+1;
      }
   }

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

   PAPIERROR("Number could not be parsed from %s",buf);
   return(-1);
}


int _ia64_get_dmem_info(PAPI_dmem_info_t *d)
{
  char fn[PATH_MAX], tmp[PATH_MAX];
  FILE *f;
  int ret;
  long long sz = 0, lck = 0, res = 0, shr = 0, stk = 0, txt = 0, dat = 0, dum = 0, lib = 0, hwm = 0;

  sprintf(fn,"/proc/%ld/status",(long)getpid());
  f = fopen(fn,"r");
  if (f == NULL)
    {
      PAPIERROR("fopen(%s): %s\n",fn,strerror(errno));
      return PAPI_ESBSTR;
    }
  while (1)
    {
      if (fgets(tmp,PATH_MAX,f) == NULL)
	break;
      if (strspn(tmp,"VmSize:") == strlen("VmSize:"))
	{
	  sscanf(tmp+strlen("VmSize:"),"%lld",&sz);
	  d->size = sz;
	  continue;
	}
      if (strspn(tmp,"VmHWM:") == strlen("VmHWM:"))
	{
	  sscanf(tmp+strlen("VmHWM:"),"%lld",&hwm);
	  d->high_water_mark = hwm;
	  continue;
	}
      if (strspn(tmp,"VmLck:") == strlen("VmLck:"))
	{
	  sscanf(tmp+strlen("VmLck:"),"%lld",&lck);
	  d->locked = lck;
	  continue;
	}
      if (strspn(tmp,"VmRSS:") == strlen("VmRSS:"))
	{
	  sscanf(tmp+strlen("VmRSS:"),"%lld",&res);
	  d->resident = res;
	  continue;
	}
      if (strspn(tmp,"VmData:") == strlen("VmData:"))
	{
	  sscanf(tmp+strlen("VmData:"),"%lld",&dat);
	  d->heap = dat;
	  continue;
	}
      if (strspn(tmp,"VmStk:") == strlen("VmStk:"))
	{
	  sscanf(tmp+strlen("VmStk:"),"%lld",&stk);
	  d->stack = stk;
	  continue;
	}
      if (strspn(tmp,"VmExe:") == strlen("VmExe:"))
	{
	  sscanf(tmp+strlen("VmExe:"),"%lld",&txt);
	  d->text = txt;
	  continue;
	}
      if (strspn(tmp,"VmLib:") == strlen("VmLib:"))
	{
	  sscanf(tmp+strlen("VmLib:"),"%lld",&lib);
	  d->library = lib;
	  continue;
	}
    }
  fclose(f);

  sprintf(fn,"/proc/%ld/statm",(long)getpid());
  f = fopen(fn,"r");
  if (f == NULL)
    {
      PAPIERROR("fopen(%s): %s\n",fn,strerror(errno));
      return PAPI_ESBSTR;
    }
  ret = fscanf(f,"%lld %lld %lld %lld %lld %lld %lld",&dum,&dum,&shr,&dum,&dum,&dat,&dum);
  if (ret != 7)
    {
      PAPIERROR("fscanf(7 items): %d\n",ret);
      return PAPI_ESBSTR;
    }
  d->pagesize = getpagesize();
  d->shared = (shr * d->pagesize)/1024;
  fclose(f);

  return PAPI_OK;
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

/*
* File:    linux-ia64-memory.c
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
#include <string.h>

inline void get_cpu_info(unsigned int *rev, unsigned int *model, unsigned int *family, unsigned int *archrev);
void fline ( FILE *fp, char *buf );
int get_number(char *buf);

#define C_INSTRUCTION 1
#define C_DATA        2
#define C_COMBINED    3

int _papi_hwd_get_memory_info(PAPI_hw_info_t * mem_info, int cpu_type)
{
   unsigned int rev,model,family,archrev;
   int retval = 0;
   FILE *f;
   int clevel=0, ctype=0;
   char buf[1024];
   int num;

   f = fopen("/proc/pal/cpu0/cache_info","r");

   if (!f)
      error_return(PAPI_ESYS, "fopen(/proc/pal/cpu0/cache_info returned < 0");

   while (!feof(f)) {
        fline(f, buf);
        if ( buf[0] == '\0' ) break;
        if (  !strncmp(buf, "Data Cache", 10) ) {
  	   ctype = C_DATA;
	   clevel = get_number( buf );
        }
        else if ( !strncmp(buf, "Instruction Cache", 17) ){
   	   ctype = C_INSTRUCTION;
	   clevel = get_number( buf );
        }
	else if ( !strncmp(buf, "Data/Instruction Cache", 22)){
           ctype = C_COMBINED;
	   clevel = get_number( buf );
        }
        else {
           if ( (clevel == 0 || clevel > 3) && ctype != 0)
	     error_return(PAPI_EBUG, "Cache type could not be recognized, send /proc/pal/cpu0/cache_info");
           if ( !strncmp(buf, "Size", 4) ){
		num = get_number( buf );
		switch(clevel){
		  case 1:
		    if ( ctype == C_INSTRUCTION ){
			num = num/1024;
       			mem_info->L1_size += num;
       			mem_info->L1_icache_size = num;
		    }
		    else if ( ctype == C_DATA ){
			num = num/1024;
       			mem_info->L1_size += num;
       			mem_info->L1_dcache_size = num;
		    }
		    else if ( ctype == C_COMBINED ){
			num = num/1024;
       			mem_info->L1_size = num;
       			mem_info->L1_dcache_size = 0;
       			mem_info->L1_icache_size = 0;
		    }
		    break;
		  case 2:
			mem_info->L2_cache_size = num/1024;
		    break;
		  case 3:
			mem_info->L3_cache_size = num/1024;
		    break;
		  default:
		    break;
		}
           }
  	   else if ( !strncmp(buf, "Associativity", 13) ) {
	   	num = get_number( buf );
		switch(clevel){
		  case 1:
		    if ( ctype == C_INSTRUCTION ){
       			mem_info->L1_icache_assoc = num;
		    }
		    else if ( ctype == C_DATA ){
       			mem_info->L1_dcache_assoc = num;
		    }
		    else if ( ctype == C_COMBINED ){
       			mem_info->L1_icache_assoc = num;
       			mem_info->L1_dcache_assoc = num;
		    }
		    break;
		  case 2:
			mem_info->L2_cache_assoc = num;
		    break;
		  case 3:
			mem_info->L3_cache_assoc = num;
		    break;
		  default:
		    break;
		}
	   }
	   else if ( !strncmp(buf, "Line size", 9) ) {
	   	num = get_number( buf );
		switch(clevel){
		  case 1:
		    if ( ctype == C_INSTRUCTION ){
		        mem_info->L1_icache_linesize = num;
			mem_info->L1_icache_lines = (mem_info->L1_icache_size*1024)/num;
		    }
		    else if ( ctype == C_DATA ){
		        mem_info->L1_dcache_linesize = num;
			mem_info->L1_dcache_lines = (mem_info->L1_icache_size*1024)/num;
		    }
		    else if ( ctype == C_COMBINED ){
		        mem_info->L1_icache_linesize = num;
		        mem_info->L1_dcache_linesize = num;
			mem_info->L1_icache_lines = (mem_info->L1_icache_size*1024)/num;
			mem_info->L1_dcache_lines = (mem_info->L1_icache_size*1024)/num;
		    }
		    break;
		  case 2:
			mem_info->L2_cache_linesize = num;
			mem_info->L2_cache_lines = (mem_info->L2_cache_size*1024)/num;
		    break;
		  case 3:
			mem_info->L3_cache_linesize = num;
			mem_info->L3_cache_lines = (mem_info->L2_cache_size*1024)/num;
		    break;
		  default:
		    break;
		}
	   }
	   else
	     continue;
        }
   } 

   get_cpu_info(&rev,&model,&family,&archrev);

   return retval;
}

int get_number( char *buf ){
   char numbers[] = "0123456789";
   int num;
   char *tmp, *end;

printf("Buf: %s\n", buf);
   tmp = strpbrk(buf, numbers);
printf("%s:%s\n", tmp, buf);
   if ( tmp != NULL ){
	end = tmp;
	while(isdigit(*end)) end++;
	*end='\0';
        num = atoi(tmp);
printf("Num: %d\n", num);
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


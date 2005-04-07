/* Compile me with -O0 or else you'll get none. */

#include "papi_test.h"

int buf[CACHE_FLUSH_BUFFER_SIZE_INTS];
int buf_dummy = 0;
volatile int *flush = NULL;
volatile int flush_dummy = 0;
volatile double a = 0.5, b = 2.2;

void do_reads(int n)
{
#if !defined(_WIN32)
   int i, retval;
   static int fd = -1;
   char buf;

   if (fd == -1) {
#if (defined(_CRAYT3E) || defined(__CATAMOUNT__))
      fd = open("./p3_events.c", O_RDONLY);
      if (fd == -1) {
         fd = open("../p3_events.c", O_RDONLY);
         if (fd == -1) {
            perror("open(./p3_events.c)");
         }
      }
#else
      fd = open("/dev/zero", O_RDONLY);
      if (fd == -1) {
         perror("open(/dev/zero)");
         exit(1);
      }
#endif
   }

   for (i = 0; i < n; i++) {
      retval = read(fd, &buf, sizeof(buf));
      if (retval != sizeof(buf))
        {
#if (defined(_CRAYT3E) || defined(__CATAMOUNT__))
          if (retval < 0)
            perror("p3_events.c cannot be read");
          else
            fprintf(stderr,"p3_events.c cannot be read: only got %d bytes.\n",retval);
#else
          if (retval < 0)
            perror("/dev/zero cannot be read");
          else
            fprintf(stderr,"/dev/zero cannot be read: only got %d bytes.\n",retval);
#endif
          exit(1);
        }
   }
#endif                          /* _WIN32 */
}

void fdo_reads(int *n)
{
  do_reads(*n);
}

void fdo_reads_(int *n)
{
  do_reads(*n);
}

void fdo_reads__(int *n)
{
  do_reads(*n);
}

void FDO_READS(int *n)
{
  do_reads(*n);
}

void _FDO_READS(int *n)
{
  do_reads(*n);
}

void do_flops(int n)
{
   int i;
   double c = 0.11;

   for (i = 0; i < n; i++) {
      c += a * b;
#ifndef _CRAYT3E
      dummy((void *) &c);
#endif
   }
}

void fdo_flops(int *n)
{
  do_flops(*n);
}

void fdo_flops_(int *n)
{
  do_flops(*n);
}

void fdo_flops__(int *n)
{
  do_flops(*n);
}

void FDO_FLOPS(int *n)
{
  do_flops(*n);
}

void _FDO_FLOPS(int *n)
{
  do_flops(*n);
}

void do_misses(int n, int size)
{
  register int i, j, tmp = buf_dummy, len = size / sizeof(int);
  dummy(buf);
  for (j = 0; j < n; j++)
    {
      for (i = 0; i < len; i++)
	{
	  buf[i] = tmp;
	}
      tmp += len;
    }
  buf_dummy = tmp;
  dummy(buf);
  dummy(&buf_dummy);
}

void fdo_misses(int *n, int *size)
{
  do_misses(*n,*size);
}

void fdo_misses_(int *n, int *size)
{
  do_misses(*n,*size);
}

void fdo_misses__(int *n, int *size)
{
  do_misses(*n,*size);
}

void FDO_MISSES(int *n, int *size)
{
  do_misses(*n,*size);
}

void _FDO_MISSES(int *n, int *size)
{
  do_misses(*n,*size);
}

void do_both(int n)
{
   int i;
   const int flops = NUM_FLOPS / n;
   const int reads = NUM_READS / n;

   for (i = 0; i < n; i++) {
      do_flops(flops);
      do_reads(reads);
   }
}

void fdo_both(int *n)
{
  do_both(*n);
}

void fdo_both_(int *n)
{
  do_both(*n);
}

void fdo_both__(int *n)
{
  do_both(*n);
}

void FDO_BOTH(int *n)
{
  do_both(*n);
}

void _FDO_BOTH(int *n)
{
  do_both(*n);
}

void do_flush(void)
{
  register int i;
  if (flush == NULL)
    flush = (int *)malloc((1024*1024*16)*sizeof(int));
  if ( !flush ) return;

  dummy((void *)flush);
  for (i=0;i<(1024*1024*16);i++)
    {
      flush[i] += flush_dummy;
    }
  flush_dummy++;
  dummy((void *)flush);
  dummy((void *)&flush_dummy);
}

void fdo_flush(void)
{
  do_flush();
}

void fdo_flush_(void)
{
  do_flush();
}

void fdo_flush__(void)
{
  do_flush();
}

void FDO_FLUSH(void)
{
  do_flush();
}

void _FDO_FLUSH(void)
{
  do_flush();
}

void do_l1misses(int n)
{
  do_misses(n,L1_MISS_BUFFER_SIZE_INTS);
}

void fdo_l1misses(int *n)
{
  do_l1misses(*n);
}

void fdo_l1misses_(int *n)
{
  do_l1misses(*n);
}

void fdo_l1misses__(int *n)
{
  do_l1misses(*n);
}

void FDO_L1MISSES(int *n)
{
  do_l1misses(*n);
}

void _FDO_L1MISSES(int *n)
{
  do_l1misses(*n);
}

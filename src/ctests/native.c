#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "test_utils.h"
#if defined(__ALPHA) && defined(__osf__)
#include <machine/hal/cpuconf.h>
#include <sys/pfcntr.h>
long get_instr()
{
  int cpu;
  GET_CPU_FAMILY(&cpu);
  if (cpu == EV6_CPU)
    return(PF6_MUX0_RET_INSTRUCTIONS << 8 | 0);
  else
    abort();
}
long get_cyc()
{
  int cpu;
  GET_CPU_FAMILY(&cpu);
  if (cpu == EV6_CPU)
    return(PF6_MUX1_CYCLES << 8 | 1);
  else
    abort();
}
#endif

static int point = 0;
static int EventSet = PAPI_NULL;
static long long us;
static const PAPI_hw_info_t *hwinfo;
extern int TESTS_QUIET; /* Declared in test_utils.c */

void papimon_start(void)
{
  int retval;
  int native;

  if (EventSet == PAPI_NULL)
    {
      if( (retval = PAPI_create_eventset(&EventSet)) != PAPI_OK )
	test_fail(__FILE__,__LINE__,"PAPI_create_eventset",retval);

#if defined(_AIX)
      native = 0 | 5 << 8  | 0; /* ICM */
      if ( (retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 35 << 8 | 1; /* FPU1CMPL */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 5 << 8  | 2; /* LDCM */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 5 << 8  | 3; /* LDCMPL */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 5 << 8  | 4; /* FPU0CMPL */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 12 << 8 | 5; /* CYC */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 9 << 8  | 6; /* FMA */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 0 << 8  | 7; /* TLB */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#elif defined(linux) && defined(__i386__)
      native = 0 | 0x43 << 8 | 0; /* Data mem refs */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 0x47 << 8 | 1; /* Lines out */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#elif defined(linux) && defined(__ia64__)
      {
	typedef union {
	  unsigned int  papi_native_all;	/* integer encoding */
	  struct	{
	    unsigned int register_no:8;	/* 4, 5, 6 or 7 */
	    unsigned int pme_mcode:8;	/* major event code */
	    unsigned int pme_ear:1;		/* is EAR event */
	    unsigned int pme_dear:1;	/* 1=Data 0=Instr */
	    unsigned int pme_tlb:1;		/* 1=TLB 0=Cache */
	    unsigned int pme_umask:13;	/* unit mask */
	  } papi_native_bits;
	} papi_native_code_t;

	/* Execution latency stall cycles */
	papi_native_code_t real_native;
	real_native.papi_native_all = 0;
	real_native.papi_native_bits.register_no = 4;
	real_native.papi_native_bits.pme_mcode = 0x02;
	native = real_native.papi_native_all;
        if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
	/* Combined execution stall cycles */
	real_native.papi_native_all = 0;
	real_native.papi_native_bits.register_no = 5;
	real_native.papi_native_bits.pme_mcode = 0x06;
	native = real_native.papi_native_all;
        if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
	/* Combined instruction fetch stall cycles */
	real_native.papi_native_all = 0;
	real_native.papi_native_bits.register_no = 6;
	real_native.papi_native_bits.pme_mcode = 0x05;
	native = real_native.papi_native_all;
        if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
	/* Combined memory stall cycles */
	real_native.papi_native_all = 0;
	real_native.papi_native_bits.register_no = 7;
	real_native.papi_native_bits.pme_mcode = 0x07;
	native = real_native.papi_native_all;
        if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      }
#elif defined(mips) && defined(sgi) && defined(unix)
      native = 0 | 0x9 << 8 | 0; /* L1 I Miss */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 0x9 << 8 | 1; /* L1 D Miss */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#elif defined(_CRAYT3E)
      native = 0 | 0x0 << 8 | 0; /* Machine cyc */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 0xe << 8 | 1; /* Dcache acc. */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 0 | 0xC << 8 | 2; /* CPU cyc */
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#elif defined(sun) && defined(sparc)
/* 
0,0,Cycle_cnt,0x0
0,0,Instr_cnt,0x1
0,0,Dispatch0_IC_miss,0x2
0,0,IC_ref,0x8
0,0,DC_rd,0x9
0,0,DC_wr,0xa
0,0,EC_ref,0xc
0,0,EC_snoop_inv,0xe
0,0,Dispatch0_storeBuf,0x3
0,0,Load_use,0xb
0,0,EC_write_hit_RDO,0xd
0,0,EC_rd_hit,0xf
0,1,Cycle_cnt,0x0
0,1,Instr_cnt,0x1
0,1,Dispatch0_mispred,0x2
0,1,EC_wb,0xd
0,1,EC_snoop_cb,0xe
0,1,Dispatch0_FP_use,0x3
0,1,IC_hit,0x8
0,1,DC_rd_hit,0x9
0,1,DC_wr_hit,0xa
0,1,Load_use_RAW,0xb
0,1,EC_hit,0xc
0,1,EC_ic_hit,0xf
*/
      native = 0 | (0xb << 8); /* Load_use */  
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = 1 | (0xa << 8); /* DC_wr_hit */  
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#elif defined(__ALPHA) && defined(__osf__)
      native = get_instr();
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
      native = get_cyc();
      if((retval = PAPI_add_event(&EventSet, native))!=PAPI_OK)
	  test_fail(__FILE__,__LINE__,"PAPI_add_event",retval);
#else
#error "Architecture not included in this test file yet."
#endif
      if ((hwinfo = PAPI_get_hardware_info()) == NULL)
	  test_fail(__FILE__,__LINE__,"PAPI_get_hardware_info",retval);
    }

  us = PAPI_get_real_usec();
  if ((retval = PAPI_start(EventSet))!=PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_start",retval);
  point++;
}

void papimon_stop(void)
{
  int retval;
  long long values[8];
  float rsec;
#if defined(_AIX)
  float csec;
#endif

  if((retval = PAPI_stop(EventSet,values))!=PAPI_OK)
     test_fail(__FILE__,__LINE__,"PAPI_stop",retval);
  us = PAPI_get_real_usec() - us;
  
  if ( !TESTS_QUIET ){
     fprintf(stderr,"-------------Monitor Point %d-------------\n",point);
     rsec = (float)us/1000000.0;
     fprintf(stderr,"Real Elapsed Time in sec.  : %f\n",rsec);
#if defined(_AIX)
     csec = (float)values[5]/(hwinfo->mhz*1000000.0);
     fprintf(stderr,"CPU Elapsed Time in sec.   : %f\n",csec);
     fprintf(stderr,"L1 Instruction Cache Misses: %lld\n",values[0]);
     fprintf(stderr,"FPU1 Instructions          : %lld\n",values[1]);
     fprintf(stderr,"L1 Data Cache Load Misses  : %lld\n",values[2]);
     fprintf(stderr,"Load Instructions          : %lld\n",values[3]);
     fprintf(stderr,"FPU0 Instructions          : %lld\n",values[4]);
     fprintf(stderr,"CPU Cycles                 : %lld\n",values[5]);
     fprintf(stderr,"FMA Instructions           : %lld\n",values[6]);
     fprintf(stderr,"TLB Misses                 : %lld\n",values[7]);
     fprintf(stderr,"------------------------------------------\n");
     fprintf(stderr,"CPU MFLOPS                 : %.2f\n",
	  (((float)values[4]+(float)values[1])/1000000.0)/csec);
     fprintf(stderr,"Real MFLOPS                : %.2f\n",
	  (((float)values[4]+(float)values[1])/1000000.0)/rsec);
     fprintf(stderr,"%% L1 Load Hit Rate         : %.2f\n",
	  100.0*(1.0 - ((float)values[2]/(float)values[3])));
     fprintf(stderr,"%% FMA Instructions         : %.2f\n",
	  100.0*(float)values[6]/((float)values[1]+(float)values[4]));
#elif defined(linux) && defined(__i386__)
     fprintf(stderr,"DCU Memory references      : %lld\n",values[0]);
     fprintf(stderr,"DCU Lines out              : %lld\n",values[1]);
#elif defined(linux) && defined(__ia64__)
     fprintf(stderr,"Execution latency stall cyc         : %lld\n",values[0]);
     fprintf(stderr,"Combined execution stall cycles     : %lld\n",values[1]);
     fprintf(stderr,"Combined instr. fetch stall cycles  : %lld\n",values[2]);
     fprintf(stderr,"Combined memory stall cycles        : %lld\n",values[3]);
#elif defined(mips) && defined(sgi)
     fprintf(stderr,"L1 Instruction cache misses       : %lld\n",values[0]);
     fprintf(stderr,"L1 Data cache misses              : %lld\n",values[1]);
#elif defined(_CRAYT3E)
     fprintf(stderr,"Machine Cycles                    : %lld\n",values[0]);
     fprintf(stderr,"DCache accesses                   : %lld\n",values[1]);
     fprintf(stderr,"CPU Cycles                        : %lld\n",values[2]);
#elif defined(sun) && defined(sparc)
     fprintf(stderr,"Load_use                   : %lld\n",values[0]);
     fprintf(stderr,"DC_wr_hit                  : %lld\n",values[1]);
#elif defined(tru64)
#endif
   }
   test_pass(__FILE__,NULL,0);
}

int main(int argc, char **argv)
{
  int retval;

  tests_quiet(argc, argv); /* Set TESTS_QUIET variable */

  if ( (retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
	test_fail(__FILE__,__LINE__,"PAPI_library_init",retval);
  
  if ( !TESTS_QUIET )
    if ((retval=PAPI_set_debug(PAPI_VERB_ECONT)) != PAPI_OK)
	test_fail(__FILE__,__LINE__,"PAPI_set_debug",retval);

  papimon_start();

  sleep(1);
  do_both(1000);

  papimon_stop();
  exit(0);
}

/*  This examples show the essentials in using the PAPI low-level
    interface. The program consists of 3 examples where the work
    done over some work-loops. The example tries to illustrate
    some simple mistakes that are easily made and how a correct
    code would accomplish the same thing.

    Example 1: The total count over two work loops (Loops 1 and 2) 
    are supposed to be measured. Due to a mis-understanding of the
    semantics of the API the total count gets wrong.
    The example also illustrates that it is legal to read both
    running and stopped counters.

    Example 2: The total count over two work loops (Loops 1 and 3)
    is supposed to be measured while discarding the counts made in
    loop 2. Instead the counts in loop1 are counted twice and the
    counts in loop2 are added to the total number of counts.

    Example 3: One correct way of accomplishing the result aimed for
    in example 2.
*/

#include "papi_test.h"

void handle_error(char *file,int line,char *msg,int errcode)
{
  char errstring[PAPI_MAX_STR_LEN];

  fprintf(stderr,"%s: %d:: %s\n",file,line,msg);
  PAPI_perror(errcode,errstring,PAPI_MAX_STR_LEN);
  fprintf(stderr,"%s: %d:: %s\n",file,line,errstring);
  exit(1);
}
int main(int argc, char **argv) 
{
  int retval;
#define NUM_EVENTS 2
  long_long values[NUM_EVENTS],dummyvalues[NUM_EVENTS];
  int Events[NUM_EVENTS]={PAPI_FP_INS,PAPI_TOT_INS};
  int EventSet=PAPI_NULL;

  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
    handle_error(__FILE__,__LINE__,"PAPI_library_init",retval);


  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_create_eventset",retval);

  retval = PAPI_add_events(&EventSet,Events,NUM_EVENTS);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_add_events",retval);

  printf("\n   Incorrect usage of read and accum.\n");
  printf("   Some cycles are counted twice\n");
  retval = PAPI_start(EventSet);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_start",retval); 

  /* Loop 1*/
  do_flops(NUM_FLOPS);
  
  retval = PAPI_read(EventSet,values);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_read",retval); 
  printf(TWO12, values[0], values[1], "(Counters continuing...)\n");

  /* Loop 2*/
  do_flops(NUM_FLOPS);
  
  /* Using PAPI_accum here is incorrect. The result is that Loop 1 *
   * is being counted twice                                        */
  retval = PAPI_accum(EventSet,values);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_accum",retval); 
  printf(TWO12, values[0], values[1], "(Counters being accumulated)\n");

  /* Loop 3*/
  do_flops(NUM_FLOPS);
  
  retval = PAPI_stop(EventSet,dummyvalues);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_stop",retval); 
  
  retval = PAPI_read(EventSet,dummyvalues);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_read",retval); 
  printf(TWO12, dummyvalues[0], dummyvalues[1], "(Reading stopped counters)\n");

  printf(TWO12, values[0], values[1], "");

  printf("\n   Incorrect usage of read and accum.\n");
  printf("   Another incorrect use\n");
  retval = PAPI_start(EventSet);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_start",retval); 

  /* Loop 1*/
  do_flops(NUM_FLOPS);
  
  retval = PAPI_read(EventSet,values);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_read",retval); 
  printf(TWO12, values[0], values[1], "(Counters continuing...)\n");

  /* Loop 2*/
  /* Code that should not be counted */
  do_flops(NUM_FLOPS);
  
  retval = PAPI_read(EventSet,dummyvalues);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_read",retval); 
  printf(TWO12, dummyvalues[0], dummyvalues[1], "(Intermediate counts...)\n");

  /* Loop 3*/
  do_flops(NUM_FLOPS);
  
  /* Since PAPI_read does not reset the counters it's use above after    *
   * loop 2 is incorrect. Instead Loop1 will in effect be counted twice. *
   * and the counts in loop 2 are included in the total counts           */
  retval = PAPI_accum(EventSet,values);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_accum",retval); 
  printf(TWO12, values[0], values[1], "");

  retval = PAPI_stop(EventSet,dummyvalues);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_stop",retval); 
  
  printf("\n   Correct usage of read and accum.\n");
  printf("   PAPI_reset and PAPI_accum used to skip counting\n");
  printf("   a section of the code.\n");
  retval = PAPI_start(EventSet);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_start",retval); 

  do_flops(NUM_FLOPS);
  
  retval = PAPI_read(EventSet,values);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_read",retval); 
  printf(TWO12, values[0], values[1], "(Counters continuing)\n");

  /* Code that should not be counted */
  do_flops(NUM_FLOPS);
  
  retval = PAPI_reset(EventSet);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_reset",retval); 
  printf("%12s %12s  (Counters reset)\n","","");

  do_flops(NUM_FLOPS);
  
  retval = PAPI_accum(EventSet,values);
  if (retval != PAPI_OK)
    handle_error(__FILE__,__LINE__,"PAPI_accum",retval); 
  printf(TWO12, values[0], values[1], "");

  PAPI_shutdown();

  printf("----------------------------------\n");  
  printf("Verification: The last line in each experiment was intended\n");
  printf("to become approximately twice the value of the first line.\n");
  printf("The third case illustrates one possible way of accomplish this.\n");
  exit(0);
}

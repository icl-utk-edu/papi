/*
 * File:    linux.c
 * CVS:     $Id$
 * Author:  Kevin London
 *          london@cs.utk.edu
 * Mods:    Haihang You
 *	       you@cs.utk.edu
 * Mods:    <Your name here>
 *          <Your email here>
 */

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

papi_vectors_t _papi_frm_vectors;
papi_vectors_t *_PAPI_CURRENT_VECTOR; 

/* Prototypes */
int vec_int_ok_dummy ();
int vec_int_one_dummy();
int vec_int_dummy ();
void vec_void_dummy();
void * vec_void_star_dummy();
long_long vec_long_long_dummy();
char * vec_char_star_dummy();
long vec_long_dummy();
long_long vec_dummy_get_virt_cycles (const hwd_context_t *zero);
long_long vec_dummy_get_virt_usec (const hwd_context_t *zero);
long_long vec_dummy_get_real_usec (void);
long_long vec_dummy_get_real_cycles (void);

extern papi_vectors_t COMP_VECTOR;
#ifdef HAVE_ACPI
extern papi_vectors_t _acpi_vectors;
#endif
#ifdef HAVE_MX
extern papi_vectors_t _mx_vectors;
#endif


papi_vectors_t *_papi_component_table[] = {
  &COMP_VECTOR,
#ifdef HAVE_ACPI
  &_acpi_vectors,
#endif
#ifdef HAVE_MX
  &_mx_vectors,
#endif
  NULL
};

void _vectors_error()
{
  SUBDBG("function is not implemented in the component!\n");
  exit(PAPI_ESBSTR);
}

long_long vec_dummy_get_real_usec (void)
{
#ifdef _WIN32
	LARGE_INTEGER PerformanceCount, Frequency;
	QueryPerformanceCounter(&PerformanceCount);
	QueryPerformanceFrequency(&Frequency);
	return((PerformanceCount.QuadPart * 1000000) / Frequency.QuadPart);
#else
  struct timeval tv;
    gettimeofday(&tv,NULL);  return((tv.tv_sec * 1000000) + tv.tv_usec);
#endif
} 
        
long_long vec_dummy_get_real_cycles (void)
{
  float usec, cyc;

  usec = (float)vec_dummy_get_real_usec();
  cyc = usec * _papi_hwi_system_info.hw_info.mhz;
  return((long_long)cyc);
}

#ifdef _BGL
   #include <stdlib.h>
   #include <sys/time.h>
   #include <sys/resource.h>
#endif

long_long vec_dummy_get_virt_usec (const hwd_context_t *zero)
{
  long_long retval;
#ifdef _BGL
      struct rusage ruse;
      getrusage(RUSAGE_SELF, &ruse);
      retval = (long long)(ruse.ru_utime.tv_sec * 1000000 + ruse.ru_utime.tv_usec);
#elif _WIN32
  /* identical code is found in the windows substrate */
    HANDLE p;
    BOOL ret;
    FILETIME Creation, Exit, Kernel, User;
    long_long virt;

    p = GetCurrentProcess();
    ret = GetProcessTimes(p, &Creation, &Exit, &Kernel, &User);
    if (ret) {
	virt = (((long_long)(Kernel.dwHighDateTime + User.dwHighDateTime))<<32)
	     + Kernel.dwLowDateTime + User.dwLowDateTime;
	retval = virt/1000;
    }
    else return(PAPI_ESBSTR);
#else

  struct tms buffer;

#ifdef __CATAMOUNT__
  retval = 0;
#else
  times(&buffer);
  retval = (long long)buffer.tms_utime*(long long)(1000000/sysconf(_SC_CLK_TCK));
#endif
#endif
  return(retval);
}

long_long vec_dummy_get_virt_cycles (const hwd_context_t *zero)
{
  float usec, cyc;

  usec = (float)vec_dummy_get_virt_usec(zero);
  cyc = usec * _papi_hwi_system_info.hw_info.mhz;
  return((long_long)cyc);
}

int vec_int_ok_dummy (){
  return PAPI_OK;
}

int vec_int_one_dummy (){
  return 1;
}

int vec_int_dummy (){
  return PAPI_ESBSTR;
}

void * vec_void_star_dummy(){
  return NULL;
}

void vec_void_dummy(){
  return;
}

long_long vec_long_long_dummy(){
  return PAPI_ESBSTR;
}

char * vec_char_star_dummy(){
  return NULL;
}

long vec_long_dummy(){
  return PAPI_ESBSTR;
}

/* Copy all non-zero component values to the framework vector */
int _papi_hwi_copy_vector_table(papi_vectors_t *frm, papi_vectors_t *cmp)
{
  if ( !frm || !cmp ) return (PAPI_EINVAL);

  /* sizes of component-private structures */
  frm->context_size =		cmp->context_size;
  frm->control_state_size =	cmp->control_state_size;
  frm->register_size =		cmp->register_size;
  frm->reg_alloc_size =		cmp->reg_alloc_size;
/*
  printf("Component -- context: %d; control_state: %d; register: %d; reg_alloc: %d\n",
      cmp->context_size, cmp->control_state_size, cmp->register_size, cmp->reg_alloc_size);
  printf("Framework -- context: %d; control_state: %d; register: %d; reg_alloc: %d\n",
      frm->context_size, frm->control_state_size, frm->register_size, frm->reg_alloc_size);
*/

  /* component function pointers */
#ifdef _WIN32 /* Windows requires a different callback format */
  if(cmp->timer_callback)	frm->timer_callback = cmp->timer_callback;
#else
  if(cmp->dispatch_timer)	frm->dispatch_timer = cmp->dispatch_timer;
#endif
  if(cmp->get_overflow_address)	frm->get_overflow_address = cmp->get_overflow_address;
  if(cmp->start)		frm->start = cmp->start;
  if(cmp->stop)			frm->stop = cmp->stop;
  if(cmp->read)			frm->read = cmp->read;
  if(cmp->reset)		frm->reset = cmp->reset;
  if(cmp->write)		frm->write = cmp->write;
  if(cmp->get_real_cycles)	frm->get_real_cycles = cmp->get_real_cycles;
  if(cmp->get_real_usec)	frm->get_real_usec = cmp->get_real_usec;
  if(cmp->get_virt_cycles)	frm->get_virt_cycles = cmp->get_virt_cycles;
  if(cmp->get_virt_usec)	frm->get_virt_usec = cmp->get_virt_usec;
  if(cmp->stop_profiling)	frm->stop_profiling = cmp->stop_profiling;
  if(cmp->init)			frm->init = cmp->init;
  if(cmp->init_control_state)	frm->init_control_state = cmp->init_control_state;
  if(cmp->update_shlib_info)	frm->update_shlib_info = cmp->update_shlib_info;
  if(cmp->get_system_info)	frm->get_system_info = cmp->get_system_info;
  if(cmp->get_memory_info)	frm->get_memory_info = cmp->get_memory_info;
  if(cmp->update_control_state)	frm->update_control_state = cmp->update_control_state;
  if(cmp->ctl)			frm->ctl = cmp->ctl;
  if(cmp->set_overflow)		frm->set_overflow = cmp->set_overflow;
  if(cmp->set_profile)		frm->set_profile = cmp->set_profile;
  if(cmp->add_prog_event)	frm->add_prog_event = cmp->add_prog_event;
  if(cmp->set_domain)		frm->set_domain = cmp->set_domain;
  if(cmp->ntv_enum_events)	frm->ntv_enum_events = cmp->ntv_enum_events;
  if(cmp->ntv_code_to_name)	frm->ntv_code_to_name = cmp->ntv_code_to_name;
  if(cmp->ntv_code_to_descr)	frm->ntv_code_to_descr = cmp->ntv_code_to_descr;
  if(cmp->ntv_code_to_bits)	frm->ntv_code_to_bits = cmp->ntv_code_to_bits;
  if(cmp->ntv_bits_to_info)	frm->ntv_bits_to_info = cmp->ntv_bits_to_info;
  if(cmp->allocate_registers)	frm->allocate_registers = cmp->allocate_registers;
  if(cmp->bpt_map_avail)	frm->bpt_map_avail = cmp->bpt_map_avail;
  if(cmp->bpt_map_set)		frm->bpt_map_set = cmp->bpt_map_set;
  if(cmp->bpt_map_exclusive)	frm->bpt_map_exclusive = cmp->bpt_map_exclusive;
  if(cmp->bpt_map_shared)	frm->bpt_map_shared = cmp->bpt_map_shared;
  if(cmp->bpt_map_preempt)	frm->bpt_map_preempt = cmp->bpt_map_preempt;
  if(cmp->bpt_map_update)	frm->bpt_map_update = cmp->bpt_map_update;
  if(cmp->get_dmem_info)	frm->get_dmem_info = cmp->get_dmem_info;
  if(cmp->shutdown)		frm->shutdown = cmp->shutdown;
  if(cmp->shutdown_global)	frm->shutdown_global = cmp->shutdown_global;
  if(cmp->user)			frm->user = cmp->user;
  return(PAPI_OK);
}

int _papi_hwi_initialize_vector(papi_vectors_t *frm){
 if ( !frm ) return (PAPI_EINVAL);

 /* sizes of component-private structures */
/* frm->context_size =		0;
 frm->control_state_size =	0;
 frm->register_size =		0;
 frm->reg_alloc_size =		0;
*/
 /* component function pointers */
#ifdef _WIN32 /* Windows requires a different callback format */
 if(!frm->timer_callback) frm->timer_callback =		(void (*) (UINT, UINT, DWORD, DWORD, DWORD)) vec_void_dummy;
#else
 if(!frm->dispatch_timer) frm->dispatch_timer =		(void (*)(int, siginfo_t *, void *)) vec_void_dummy;
#endif
 if(!frm->get_overflow_address) frm->get_overflow_address=	(void *(*) (int, char *)) vec_void_star_dummy;
 if(!frm->start) frm->start=			(int (*) (hwd_context_t *, hwd_control_state_t *)) vec_int_dummy;
 if(!frm->stop) frm->stop=			(int (*) (hwd_context_t *, hwd_control_state_t *)) vec_int_dummy;
 if(!frm->read) frm->read=			(int (*)(hwd_context_t *, hwd_control_state_t *, long_long **, int)) vec_int_dummy;
 if(!frm->reset) frm->reset =			(int (*) (hwd_context_t *, hwd_control_state_t *)) vec_int_dummy;
 if(!frm->write) frm->write=			(int (*) (hwd_context_t *, hwd_control_state_t *, long_long[])) vec_int_dummy;
 if(!frm->get_real_cycles) frm->get_real_cycles=		(long_long (*) ()) vec_dummy_get_real_cycles;
 if(!frm->get_real_usec) frm->get_real_usec=		(long_long (*) ()) vec_dummy_get_real_usec;
 if(!frm->get_virt_cycles) frm->get_virt_cycles=		vec_dummy_get_virt_cycles;
 if(!frm->get_virt_usec) frm->get_virt_usec=		vec_dummy_get_virt_usec;
 if(!frm->stop_profiling) frm->stop_profiling=		(int (*) (ThreadInfo_t *, EventSetInfo_t *)) vec_int_dummy;
 if(!frm->init) frm->init=			(int (*) (hwd_context_t *)) vec_int_ok_dummy;
 if(!frm->init_control_state) frm->init_control_state=	(int (*) (hwd_control_state_t * ptr)) vec_void_dummy;
 if(!frm->update_shlib_info) frm->update_shlib_info=	(int (*) (void)) vec_int_dummy;
 if(!frm->get_system_info) frm->get_system_info=		(int (*) ()) vec_int_dummy;
 if(!frm->get_memory_info) frm->get_memory_info=		(int (*) (PAPI_hw_info_t *, int)) vec_int_dummy;
 if(!frm->update_control_state) frm->update_control_state=	(int (*) (hwd_control_state_t *, NativeInfo_t *, int, hwd_context_t *)) vec_int_dummy;
 if(!frm->ctl) frm->ctl=			(int (*) (hwd_context_t *, int, _papi_int_option_t *)) vec_int_dummy;
 if(!frm->set_overflow) frm->set_overflow=		(int (*) (EventSetInfo_t *, int, int)) vec_int_dummy;
 if(!frm->set_profile) frm->set_profile=		(int (*) (EventSetInfo_t *, int, int)) vec_int_dummy;
 if(!frm->add_prog_event) frm->add_prog_event=		(int (*) (hwd_control_state_t *, unsigned int, void *, EventInfo_t *)) vec_int_dummy;
 if(!frm->set_domain) frm->set_domain=		(int (*) (hwd_control_state_t *, int)) vec_int_dummy;
 if(!frm->ntv_enum_events) frm->ntv_enum_events=		(int (*) (unsigned int *, int)) vec_int_dummy;
 if(!frm->ntv_code_to_name) frm->ntv_code_to_name=	(char * (*) (unsigned int)) vec_char_star_dummy;
 if(!frm->ntv_code_to_descr) frm->ntv_code_to_descr=	(char * (*) (unsigned int)) vec_char_star_dummy;
 if(!frm->ntv_code_to_bits) frm->ntv_code_to_bits=	(int (*) (unsigned int, hwd_register_t *)) vec_int_dummy;
 if(!frm->ntv_bits_to_info) frm->ntv_bits_to_info=	(int (*) (hwd_register_t *, char *, unsigned int *, int, int)) vec_int_dummy;
 if(!frm->allocate_registers) frm->allocate_registers=	(int (*) (EventSetInfo_t *)) vec_int_one_dummy;
 if(!frm->bpt_map_avail) frm->bpt_map_avail=		(int (*) (hwd_reg_alloc_t *, int)) vec_int_dummy;
 if(!frm->bpt_map_set) frm->bpt_map_set=		(void (*) (hwd_reg_alloc_t *, int)) vec_void_dummy;
 if(!frm->bpt_map_exclusive) frm->bpt_map_exclusive=	(int (*) (hwd_reg_alloc_t *)) vec_int_dummy;
 if(!frm->bpt_map_shared) frm->bpt_map_shared=		(int (*) (hwd_reg_alloc_t *, hwd_reg_alloc_t *)) vec_int_dummy;
 if(!frm->bpt_map_preempt) frm->bpt_map_preempt=		(void (*) (hwd_reg_alloc_t *, hwd_reg_alloc_t *)) vec_void_dummy;
 if(!frm->bpt_map_update) frm->bpt_map_update=		(void (*) (hwd_reg_alloc_t *, hwd_reg_alloc_t *)) vec_void_dummy;
 if(!frm->get_dmem_info) frm->get_dmem_info=		(int (*) (PAPI_dmem_info_t *)) vec_int_dummy;
 if(!frm->shutdown) frm->shutdown=		(int (*) (hwd_context_t *)) vec_int_dummy;
 if(!frm->shutdown_global) frm->shutdown_global=		(int (*) (void)) vec_int_ok_dummy;
 if(!frm->user) frm->user=			(int (*) (int, void *, void *)) vec_int_dummy;
  return PAPI_OK;
}

int PAPI_user(int func_num, void * input, void * output){
  return (_PAPI_CURRENT_VECTOR->user(func_num, input, output));
}

char * find_dummy(void * func, char **buf){
  void * ptr=NULL;

  if ( vec_int_ok_dummy == (int (*)())func){
    ptr = ( void *)vec_int_ok_dummy;
    *buf = papi_strdup("vec_int_ok_dummy");
  }
  else if ( vec_int_one_dummy == (int (*)())func ){
    ptr = ( void *)vec_int_one_dummy;
    *buf = papi_strdup("vec_int_one_dummy");
  }
  else if ( vec_int_dummy == (int (*)())func ){
    ptr = ( void *)vec_int_dummy;
    *buf = papi_strdup("vec_int_dummy");
  }
  else if ( vec_void_dummy == (void (*)())func ){
    ptr = (void *)vec_void_dummy;
    *buf = papi_strdup("vec_void_dummy");
  }
  else if ( vec_void_star_dummy == (void *(*)())func ){
    ptr = (void *)vec_void_star_dummy;
    *buf = papi_strdup("vec_void_star_dummy");
  }
  else if ( vec_long_long_dummy == (long_long (*)())func ){
    ptr = (void *)vec_long_long_dummy;
    *buf = papi_strdup("vec_long_long_dummy");
  }
  else if ( vec_char_star_dummy == (char*(*)())func ){
    ptr = (void *)vec_char_star_dummy;
    *buf = papi_strdup("vec_char_star_dummy");
  }
  else if ( vec_long_dummy == (long (*)())func ){
    ptr = (void *)vec_long_dummy;  
    *buf = papi_strdup("vec_long_dummy");
  }
  else if ( vec_dummy_get_real_usec == (long_long(*)(void))func ) {
    ptr = (void *)vec_dummy_get_real_usec;
    *buf = papi_strdup("vec_dummy_get_real_usec");
  }
  else if ( vec_dummy_get_real_cycles == (long_long(*)(void))func ) {
    ptr = (void *)vec_dummy_get_real_cycles;
    *buf = papi_strdup("vec_dummy_get_real_cycles");
  }
  else if ( vec_dummy_get_virt_usec == (long_long(*)(const hwd_context_t *))func ) {
    ptr = (void *)vec_dummy_get_virt_usec;
    *buf = papi_strdup("vec_dummy_get_virt_usec");
  }
  else if ( vec_dummy_get_virt_cycles == (long_long(*)(const hwd_context_t *))func ) {
    ptr = (void *)vec_dummy_get_virt_cycles;
    *buf = papi_strdup("vec_dummy_get_virt_cycles");
  }
  else {
    ptr = NULL;
  }
  return(ptr);
}

void vector_print_routine( void*func, char *fname, int pfunc){
  void * ptr=NULL;
  char  *buf=NULL;

  ptr = find_dummy(func, &buf);
  if ( ptr ){
    printf("%s: %s is mapped to %s.\n", (ptr?"DUMMY":"function"),fname, buf);
    papi_free(buf);
  }
  else if ( (!ptr && pfunc) )
    printf("%s: %s is mapped to %p.\n", (ptr?"DUMMY":"function"),fname, func);
}

void vector_print_table(papi_vectors_t *frm, int print_func){

 if (!frm) return;

#ifdef _WIN32 /* Windows requires a different callback format */
 vector_print_routine((void *)frm->timer_callback, "_papi_hwd_timer_callback",print_func);
#else
 vector_print_routine((void *)frm->dispatch_timer, "_papi_hwd_dispatch_timer",print_func);
#endif
 vector_print_routine((void *)frm->get_overflow_address, "_papi_hwd_get_overflow_address",print_func);
 vector_print_routine((void *)frm->start, "_papi_hwd_start",print_func);
 vector_print_routine((void *)frm->stop, "_papi_hwd_stop",print_func);
 vector_print_routine((void *)frm->read,"_papi_hwd_read",print_func);
 vector_print_routine((void *)frm->reset, "_papi_hwd_reset",print_func);
 vector_print_routine((void *)frm->write, "_papi_hwd_write",print_func);
 vector_print_routine((void *)frm->get_real_cycles, "_papi_hwd_get_real_cycles",print_func);
 vector_print_routine((void *)frm->get_real_usec, "_papi_hwd_get_real_usec",print_func);
 vector_print_routine((void *)frm->get_virt_cycles, "_papi_hwd_get_virt_cycles",print_func);
 vector_print_routine((void *)frm->get_virt_usec, "_papi_hwd_get_virt_usec",print_func);
 vector_print_routine((void *)frm->stop_profiling, "_papi_hwd_stop_profiling",print_func);
 vector_print_routine((void *)frm->init, "_papi_hwd_init",print_func);
 vector_print_routine((void *)frm->init_control_state, "_papi_hwd_init_control_state",print_func);
 vector_print_routine((void *)frm->ctl, "_papi_hwd_ctl",print_func);
 vector_print_routine((void *)frm->set_overflow, "_papi_hwd_set_overflow",print_func);
 vector_print_routine((void *)frm->set_profile, "_papi_hwd_set_profile",print_func);
 vector_print_routine((void *)frm->add_prog_event, "_papi_hwd_add_prog_event",print_func);
 vector_print_routine((void *)frm->set_domain, "_papi_hwd_set_domain",print_func);
 vector_print_routine((void *)frm->ntv_enum_events, "_papi_hwd_ntv_enum_events",print_func);
 vector_print_routine((void *)frm->ntv_code_to_name, "_papi_hwd_ntv_code_to_name",print_func);
 vector_print_routine((void *)frm->ntv_code_to_descr, "_papi_hwd_ntv_code_to_descr",print_func);
 vector_print_routine((void *)frm->ntv_code_to_bits, "_papi_hwd_ntv_code_to_bits",print_func);
 vector_print_routine((void *)frm->ntv_bits_to_info, "_papi_hwd_ntv_bits_to_info",print_func);
 vector_print_routine((void *)frm->allocate_registers, "_papi_hwd_allocate_registers",print_func);
 vector_print_routine((void *)frm->bpt_map_avail, "_papi_hwd_bpt_map_avail",print_func);
 vector_print_routine((void *)frm->bpt_map_set, "_papi_hwd_bpt_map_set",print_func);
 vector_print_routine((void *)frm->bpt_map_exclusive, "_papi_hwd_bpt_map_exclusive",print_func);
 vector_print_routine((void *)frm->bpt_map_shared, "_papi_hwd_bpt_shared",print_func);
 vector_print_routine((void *)frm->bpt_map_update, "_papi_hwd_bpt_map_update",print_func);
 vector_print_routine((void *)frm->get_dmem_info, "_papi_hwd_get_dmem_info",print_func);
 vector_print_routine((void *)frm->shutdown, "_papi_hwd_shutdown",print_func);
 vector_print_routine((void *)frm->shutdown_global, "_papi_hwd_shutdown_global",print_func);
 vector_print_routine((void *)frm->user, "_papi_hwd_user",print_func);
}

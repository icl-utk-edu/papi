#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
  #include <unistd.h>
#endif
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#undef NDEBUG
#include <malloc.h>
#include "papiStdEventDefs.h"


#ifndef _WIN32
  #include "papi.h"
  #include "papi_internal.h"

  int main(int argc, char **argv)
  {
    exit(Nils_System_Info());
  }
#else
  #include "win32.h"
#endif

int Nils_System_Info(void) 
{
  int i;
  const PAPI_preset_info_t *info = NULL;
  const PAPI_hw_info_t *hwinfo = NULL;

  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    return(1);

  if (PAPI_set_debug(PAPI_VERB_ECONT) != PAPI_OK)
    return(1);

  if ((info = PAPI_query_all_events_verbose()) == NULL)
    return(1);

  if ((hwinfo = PAPI_get_hardware_info()) == NULL)
    return(1);

  printf("Test case 8: Available events and hardware information.\n");
  printf("-------------------------------------------------------------------------\n");
  printf("Vendor string and code   : %s (%d)\n",hwinfo->vendor_string,hwinfo->vendor);
  printf("Model string and code    : %s (%d)\n",hwinfo->model_string,hwinfo->model);
  printf("CPU revision             : %f\n",hwinfo->revision);
  printf("CPU Megahertz            : %f\n",hwinfo->mhz);
  printf("CPU's in an SMP node     : %d\n",hwinfo->ncpu);
  printf("Nodes in the system      : %d\n",hwinfo->nnodes);
  printf("Total CPU's in the system: %d\n",hwinfo->totalcpus);
  printf("-------------------------------------------------------------------------\n");
  /* This is not recommended way to use the PAPI library, but it does give
     us some nice info... */
#define _psi _papi_system_info
#define YESNO(flag) ( flag == -1 ? "-1 ???" : ( flag ? "YES" : "NO" ) )

  printf("\n");
  printf("Substrate name (version) : %s (%f)\n",_psi.substrate,_psi.version);
  printf("Number of counters returned by substrate: %d\n",_psi.num_cntrs);
  printf("Substrate supports counter groups: %s\n",YESNO(_psi.grouped_counters));
  if(_psi.grouped_counters > 0)
    printf("    Number of groups: %d\n",_psi.grouped_counters);

  printf("Number of general purpose counters or counters per group: %d %s\n",
	 _psi.num_gp_cntrs,"  What???");

  printf("Number of special purpose counters: %d\n",_psi.num_sp_cntrs);
  printf("Number of preset events: %d\n",_psi.total_presets);
  printf("Number of native events: %d\n",_psi.total_events);
  printf("Initial default domain: 0x%.8x\n",_psi.default_domain);
  printf("    ");
  if (_psi.default_domain & PAPI_DOM_USER   ) printf(" %s","PAPI_DOM_USER");
  if (_psi.default_domain & PAPI_DOM_MIN    ) printf(" %s","PAPI_DOM_MIN");
  if (_psi.default_domain & PAPI_DOM_KERNEL ) printf(" %s","PAPI_DOM_KERNEL");
  if (_psi.default_domain & PAPI_DOM_OTHER  ) printf(" %s","PAPI_DOM_OTHER");
  if (_psi.default_domain & PAPI_DOM_ALL    ) printf(" %s","PAPI_DOM_ALL");
  if (_psi.default_domain & PAPI_DOM_MAX    ) printf(" %s","PAPI_DOM_MAX");
  if (_psi.default_domain & PAPI_DOM_HWSPEC ) printf(" %s","PAPI_DOM_HWSPEC");
  printf("\n");

  printf("Initial default granularity: 0x%.8x\n",_psi.default_granularity);
  printf("    ");
  if (_psi.default_granularity & PAPI_GRN_THR    ) printf(" %s","PAPI_GRN_THR");
  if (_psi.default_granularity & PAPI_GRN_MIN    ) printf(" %s","PAPI_GRN_MIN");
  if (_psi.default_granularity & PAPI_GRN_PROC   ) printf(" %s","PAPI_GRN_PROC");
  if (_psi.default_granularity & PAPI_GRN_PROCG  ) printf(" %s","PAPI_GRN_PROCG");
  if (_psi.default_granularity & PAPI_GRN_SYS    ) printf(" %s","PAPI_GRN_SYS");
  if (_psi.default_granularity & PAPI_GRN_SYS_CPU) printf(" %s","PAPI_GRN_SYS_CPU");
  if (_psi.default_granularity & PAPI_GRN_MAX    ) printf(" %s","PAPI_GRN_MAX");
  printf("\n");

  printf("%-45s %s\n","Programmable events:",YESNO(_psi.supports_program));
  printf("%-45s %s\n","Writeable counters:",YESNO(_psi.supports_write));
  printf("%-45s %s\n","Overflow supported by h/w:",YESNO(_psi.supports_hw_overflow));
  printf("%-45s %s\n","Hardware profile supported:",YESNO(_psi.supports_hw_profile));
  printf("%-45s %s\n","Full prec. (64-bit) virtual counters:",YESNO(_psi.supports_64bit_counters));
  printf("%-45s %s\n","Pass on/inheritance of child counters/values:",YESNO(_psi.supports_inheritance));
  printf("%-45s %s\n","We can attach PAPI to another process:",YESNO(_psi.supports_attach));
  printf("%-45s %s\n","We can use the real_usec call:",YESNO(_psi.supports_real_usec));
  printf("%-45s %s\n","We can use the real_cyc call:",YESNO(_psi.supports_real_cyc));
  printf("%-45s %s\n","We can use the virt_usec call:",YESNO(_psi.supports_virt_usec));
  printf("%-45s %s\n","We can use the virt_cyc call:",YESNO(_psi.supports_virt_cyc));
  printf("%-45s %s\n","Kernel read resets counters:",YESNO(_psi.supports_read_reset));

  printf("\n");

  return(0);
  /* Excerpt from papi_internal.h 

  const char substrate[81]; ;;; Name of the substrate we're using 
  const float version;      ;;; Version of this substrate 

  int num_cntrs;   ;;; Number of counters returned by a substrate read/write 
                      
  int num_gp_cntrs;   ;;; Number of general purpose counters or counters
                         per group 
  int grouped_counters;   ;;; Number of counter groups, zero for no groups 
  int num_sp_cntrs;   ;;; Number of special purpose counters, like 
                         Time Stamp Counter on IBM or Pentium 

  int total_presets;  ;;; Number of preset events supported 
  int total_events;   ;;; Number of native events supported. 

  const int default_domain; ;;; The default domain when this substrate is used 
  const int default_granularity; ;;; The default granularity when this substrate is used 

  -- Begin public feature flags --

  const int supports_program;        ;;; We can use programmable events 
  const int supports_write;          ;;; We can write the counters 
  const int supports_hw_overflow;    ;;; Needs overflow to be emulated 
  const int supports_hw_profile;     ;;; Needs profile to be emulated 
  const int supports_64bit_counters; ;;; Only limited precision is available from hardware 
  const int supports_inheritance;    ;;; We can pass on and inherit child counters/values 
  const int supports_attach;         ;;; We can attach PAPI to another process 
  const int supports_real_usec;      ;;; We can use the real_usec call 
  const int supports_real_cyc;       ;;; We can use the real_cyc call 
  const int supports_virt_usec;      ;;; We can use the virt_usec call 
  const int supports_virt_cyc;       ;;; We can use the virt_cyc call                     

  -- Begin private feature flags --

  const int supports_read_reset;     ;;; The read call from the kernel resets the counters 
  */

  printf("-------------------------------------------------------------------------\n");
  printf("Name\t\tCode\t\tAvail\tDeriv\tDescription (Note)\n");
  for (i=0;i<PAPI_MAX_PRESET_EVENTS;i++)
    if (info[i].event_name)
      printf("%s\t0x%x\t%s\t%s\t%s (%s)\n",
	     info[i].event_name,
	     info[i].event_code,
	     (info[i].avail ? "Yes" : "No"),
	     (info[i].flags & PAPI_DERIVED ? "Yes" : "No"),
	     info[i].event_descr,
	     (info[i].event_note ? info[i].event_note : ""));
  printf("-------------------------------------------------------------------------\n");

  printf("Verification:\n");
  printf("Check your architecture and substrate file\n");
  
  return(0);
}

#include "papi.h"
#include "papi_internal.h"
#include "papi_protos.h"
#include "papi_vector.h"
#include "papi_memory.h"

char *  const init_str[] = {"proc_substrate" 
#ifdef HAS_ACPI
	,"acpi"
#endif
  , NULL
};

int _papi_hwd_init_acpi_substrate(papi_vectors_t *vtable, int idx);

InitPtr _papi_hwi_find_init(char *name){

  if ( name == NULL ) return NULL;

  if (!strcmp(name, "proc_substrate")) return _papi_hwd_init_substrate;
#ifdef HAS_ACPI
  if ( !strcmp(name, "acpi")) return _papi_hwd_init_acpi_substrate;
#endif

  INTDBG("No substrate named: %s found!\n", name);

  return NULL;
}

#include "papi.h"
#include SUBSTRATE


int get_memory_info(PAPI_hw_info_t * mem_info)
{
   return PAPI_OK;
}

long _papi_hwd_get_dmem_info(int option)
{
   switch (option) {
   case PAPI_GET_RESSIZE:
      return (1);
   case PAPI_GET_SIZE:
      return (2);
   default:
      return (PAPI_OK);
   }
}

C This file performs the following tests:
C Hardware info

#include "fpapi.h"

      program avail
      INTEGER ncpu,nnodes,totalcpus,vendor,model, check, handle
      CHARACTER*20 vstring, mstring
      REAL revision, mhz

      handle=0 
      call PAPIf_library_init(check)
      if ( check.NE.PAPI_VER_CURRENT) then
         print *, 'Error in PAPI_library_init '
         stop
      end if
      call PAPIf_get_hardware_info( ncpu,nnodes,totalcpus,vendor,
     .     vstring, model, mstring, revision, mhz )

      print *, 'Test case 8: Hardware information'
      print *, '------------------------------------------------'
      print *, 'Vendor string and code   : ',vstring,' (',vendor,')'
      print *, 'Model string and code    : ',mstring,' (',model,')'
      print *, 'CPU revision             : ',revision 
      print *, 'CPU Megahertz            : ',mhz
      print *, 'CPUs in an SMP node     : ',ncpu
      print *, 'Nodes in the system      : ',nnodes
      print *, 'Total CPUs in the system: ',totalcpus
      print *, '------------------------------------------------'
      end

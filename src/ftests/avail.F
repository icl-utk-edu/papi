C This file performs the following tests:
C Hardware info
#include "ftests.h"
#include "fpapi.h"

      program avail
      IMPLICIT NONE

      INTEGER ncpu,nnodes,totalcpus,vendor,model, check, handle, n
      CHARACTER*(PAPI_MAX_STR_LEN) vstring, mstring
      REAL revision, mhz
      integer len_trim
      external len_trim

      handle=0 
      check = PAPI_VER_CURRENT
      call PAPIf_library_init(check)
      if ( check.NE.PAPI_VER_CURRENT) then
         FATAL_ERROR('Error in PAPI_library_init')
      end if
      call PAPIf_get_hardware_info( ncpu,nnodes,totalcpus,vendor,
     .     vstring, model, mstring, revision, mhz )

      print *, 'Test case 8: Hardware information'
      print *, '------------------------------------------------'
      n=len_trim(vstring)
      print *, 'Vendor string and code   : ',vstring(1:n)
     & ,' (',vendor,')'
      n=len_trim(mstring)
      print *, 'Model string and code    : ',mstring(1:n),' (',model,')'
      print *, 'CPU revision             : ',revision 
      print *, 'CPU Megahertz            : ',mhz
      print *, 'CPUs in an SMP node      : ',ncpu
      print *, 'Nodes in the system      : ',nnodes
      print *, 'Total CPUs in the system : ',totalcpus
      print *, '------------------------------------------------'
      end

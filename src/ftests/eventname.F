#include "fpapi.h"

      program eventname

      INTEGER retval, handle
      INTEGER preset

      handle = 0
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
         print *, 'Error in PAPI_library_init '
         stop
      end if

      call PAPIf_event_name_to_code( 'PAPI_FP_INS',preset,retval )
      if (retval.NE.PAPI_OK ) then
       print *, 'Assertion failed retval==PAPI_OK, eventname.f, line 10'
      end if
      end


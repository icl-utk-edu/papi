#include "fpapi.h"

      program eventname
      IMPLICIT NONE

      INTEGER retval, handle
      INTEGER preset

      handle = 0
      retval = PAPI_VER_CURRENT
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftests_perror(__LINE__,'Error in PAPI_library_init ')
        call ftests_fail(__FILE__,retval)
      end if

      call PAPIf_event_name_to_code( 'PAPI_FP_INS',preset,retval )
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'Event name to code')
        call ftests_fail(__FILE__,retval)
      end if

      print *, 'Ok, PAPI_FP_INS code is',preset
      call ftests_pass(__FILE__)
      end


#include "fpapi.h"

      program hello

      implicit none

      integer retval



      call PAPIf_hl_region_begin("main@", retval)

      if ( retval .NE. PAPI_OK ) then

         print *, "PAPIf_hl_region_begin failed!"

      end if



      print *, "Hello World!"



      call PAPIf_hl_region_end("main@", retval)

      if ( retval .NE. PAPI_OK ) then

         print *, "PAPIf_hl_region_end failed!"

      end if



      end program hello

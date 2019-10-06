#include "fpapi.h"

      program flops
      integer retval
      integer i

      call PAPIf_hl_init(retval)
      if ( retval .NE. PAPI_OK ) then
         write (*,*) "PAPIf_hl_init failed!"
      end if

      call PAPIf_hl_set_events("perf::TASK-CLOCK", retval)
      if ( retval .NE. PAPI_OK ) then
         write (*,*) "PAPIf_hl_set_events failed!"
      end if

      do i = 1, 4
         call PAPIf_hl_region_begin("main", retval)
         if ( retval .NE. PAPI_OK ) then
            write (*,*) "PAPIf_hl_region_begin failed!"
         end if

         write (*,*) 'Round', i
         call fdo_flops(NUM_FLOPS)

         call PAPIf_hl_region_end("main", retval)
         if ( retval .NE. PAPI_OK ) then
            write (*,*) "PAPIf_hl_region_end failed!"
         end if
      end do

      call PAPIf_hl_print_output()

      call PAPIf_hl_finalize(retval)
      if ( retval .NE. PAPI_OK ) then
         write (*,*) "PAPIf_hl_finalize failed!"
      end if

      call ftests_pass(__FILE__)
      end program flops

C This file performs the following tests:
C Hardware info
#include "fpapi.h"
#define PAPI_MAX_PRESET_EVENTS 128

      program avail
      IMPLICIT NONE

      INTEGER ncpu,nnodes,totalcpus,vendor,model, check, handle, n
      CHARACTER*(PAPI_MAX_STR_LEN) vstring, mstring
      REAL revision, mhz
      integer len_trim
      external len_trim
      integer i, avail, flags
      CHARACTER*(PAPI_MAX_STR_LEN) event_name, event_descr, 
     *event_label, event_note
      CHARACTER*(10) avail_str, flags_str

      handle=0 
      check = PAPI_VER_CURRENT
      call PAPIf_library_init(check)
      if ( check.NE.PAPI_VER_CURRENT) then
C         call ftests_fail(__FILE__'PAPI_library_init',check)
         call ftests_fail('PAPI_library_init',check)
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

      print *, 'Name\t\tCode\t\tAvail\tDeriv\tDescription(note)'
      do i=0, PAPI_MAX_PRESET_EVENTS-1
         call papif_query_all_event_verbose(z'80000000'+i, event_name, 
     *   event_descr, event_label, avail, event_note, flags, check)
         if (avail.EQ.1) then
            avail_str = 'Yes'
         else 
            avail_str = 'No'
         end if

         if (flags.EQ.1) then
            flags_str = 'Yes'
         else 
            flags_str = 'No'
         end if

         if (check.EQ.PAPI_OK) then
            write (*, 100) event_name, z'80000000'+i, avail_str, 
     *       flags_str, event_descr, event_note
 100  format(A20, '\t', Z8, '\t', A5, '\t',  A5, '\t', A50, 
     *'(', A30, ')') 
         end if
      end do
      print *, '------------------------------------------------'

      call ftests_pass(__FILE__)      
      end

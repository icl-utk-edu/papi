C From Dave McNamara at PSRV. Thanks! 
C Ported to Fortran by Kevin London
C If you try to add an event that doesn't exist, you get the correct error
C message, yet you get subsequent Seg. Faults when you try to do PAPI_start 
C and PAPI_stop. I would expect some bizarre behavior if I had no events 
C added to the event set and then tried to PAPI_start but if I had 
C successfully added one event, then the 2nd one get an error when I 
C tried to add it, is it possible for PAPI_start to work but just 
C count the first event?

#include  "fpapi.h"

      program case1
      IMPLICIT NONE

      INTEGER EventSet
      INTEGER retval
      INTEGER i,errorcode
      CHARACTER*(PAPI_MAX_STR_LEN) errorstring
      INTEGER*8 gl(2)
      INTEGER n
      REAL c,a,b

      n = 1000
      a = 0.999
      b = 1.001

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
         print *, 'Error in PAPI_library_init '
         stop
      end if

      call PAPIf_create_eventset( EventSet, retval )
      if ( retval .NE. PAPI_OK ) print *, 'error creating EventSet'

      call PAPIf_add_event( EventSet, PAPI_L2_TCM, retval )
      if ( retval .NE. PAPI_OK ) then
        print *, 'error adding PAPI_L2_TCM'
        call PAPIf_perror( retval, errorstring, errorcode)
        print *, errorstring
      else
        print *,'Added PAPI_L2_TCM'
      end if

      call PAPIf_add_event( EventSet, PAPI_L2_DCM, retval )
      if ( retval .NE. PAPI_OK )then
         print *, 'error adding PAPI_L2_DCM'
         call PAPIf_perror( retval, errorstring, errorcode)
         print *, errorstring
      else
         print *,'Added PAPI_L2_DCM'
      end if
  
      call PAPIf_start( EventSet, retval )


      do i=1, n
       c = a * b
      end do

      call PAPIf_stop( EventSet, gl, retval ) 
      end

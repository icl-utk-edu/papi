C ****************************************************************************
C
C fmatrixpapi.f
C An example of matrix-matrix multiplication and using PAPI high level to 
C look at the performance. written by Kevin London
C March 2000
C ****************************************************************************

#include "fpapi.h"

      program fmatrixpapi
      IMPLICIT NONE

      INTEGER ncols1,nrows1,ncols2,nrows2
      PARAMETER(nrows1=175,ncols1=225,nrows2=ncols1,ncols2=150)
      INTEGER i,j,k,num_events,retval
C   PAPI standardized event to be monitored
      INTEGER event(2)
C   PAPI values of the counters
      INTEGER*8 values(2)
      DOUBLE PRECISION p(nrows1,ncols1),q(nrows2,ncols2),
     &                 r(nrows1,ncols2)
      integer tests_quiet
      character*25 chbuf

      tests_quiet=0
      call getarg(1,chbuf)
      if (LGE(chbuf, 'TESTS_QUIET')) then
         tests_quiet=1
      end if

C   Setup default values
      num_events=0

C   Open matrix file number 1 for reading
C      OPEN(UNIT=1,FILE='fmt1',STATUS='OLD')
C   Open matrix file number 2 for reading
C      OPEN(UNIT=2,FILE='fmt2',STATUS='OLD')

C   Total floating point operations
      event(1) = PAPI_FP_INS
C  Time used
      event(2) = PAPI_TOT_CYC

C   See how many hardware events at one time are supported
      call PAPIf_num_counters( num_events )
      if ( num_events .LT. 2 ) then
          print *,'This example program requries the architecture ',
     .    'to support 2 simultaneous hardware events...shutting down.'
      call ftests_skip(__FILE__,retval)
      end if

      if (tests_quiet .EQ. 0) then
      print *, 'Number of hardware counters supported: ', num_events
      end if

C   matrix 1: read in the matrix values
      do i=1, nrows1
         do j=1,ncols1
            p(i,j) = i*j*1.0
         end do
      end do

C   matrix 2: read in the matrix values
      do i=1, nrows2
         do j=1,ncols2
            q(i,j) = i*j*1.0
         end do
      end do

C  Initialize the result matrix 
      do i=1,nrows1
         do j=1, ncols2
            r(i,j) = i*j*1.0
         end do
      end do
      
C  Set up the counters
      num_events = 2
      call PAPIf_start_counters( event, num_events, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'Error starting counters')
        call ftests_fail(__FILE__,retval)
      end if

C  Clear the counter values
      call PAPIf_read_counters(values, num_events,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'Error reading counters')
        call ftests_fail(__FILE__,retval)
      end if

C  Compute the matrix-matrix multiplication
      do i=1,nrows1
       do j=1,ncols2
         do k=1,ncols1
	   r(i,j)=r(i,j) + p(i,k)*q(k,j)
         end do
       end do
      end do

C  Stop the counters and put the results in the array values 
      call PAPIf_stop_counters(values,num_events,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'Error stopping counters')
        call ftests_fail(__FILE__,retval)
      end if
      call PAPIf_read_counters(values, num_events,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftests_perror(__LINE__,'Error reading counters')
        call ftests_fail(__FILE__,retval)
      end if

C  Make sure the compiler does not optimize away the multiplication
      call dummy(r)

      if (tests_quiet .EQ. 0) then
      print *, 'FP Instructions:  ',values(1)
      print *, 'Cycles: ',values(2)
      write(*,'(a,f9.6)') ' Efficiency (flops/cycles):',
     &                         real(values(1))/real(values(2))
      end if

      call ftests_pass(__FILE__)
      end

C****************************************************************************
C     
C     fmatrixpapi.f
C     An example of matrix-matrix multiplication and using PAPI high level to 
C     look at the performance. written by Kevin London
C     March 2000
C****************************************************************************

#include "fpapi_test.h"

      program fmatrixpapi
      IMPLICIT integer (p)

      INTEGER ncols1,nrows1,ncols2,nrows2
      PARAMETER(nrows1=175,ncols1=225,nrows2=ncols1,ncols2=150)
      INTEGER i,j,k,num_events,retval
C     PAPI standardized event to be monitored
      INTEGER event(2)
C     PAPI values of the counters
      INTEGER*8 values(2)
      REAL*8 p(nrows1,ncols1),q(nrows2,ncols2),
     &     r(nrows1,ncols2),tmp
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()

C     Setup default values
      num_events=0

C     Open matrix file number 1 for reading
C     OPEN(UNIT=1,FILE='fmt1',STATUS='OLD')
C     Open matrix file number 2 for reading
C     OPEN(UNIT=2,FILE='fmt2',STATUS='OLD')

C     See how many hardware events at one time are supported
C     This also initializes the PAPI library
      call PAPIf_num_counters( num_events )
      if ( num_events .LT. 2 ) then
        print *,'This example program requries the architecture to ',
     .       'support 2 simultaneous hardware events...shutting down.'
        call ftest_skip(__FILE__, __LINE__,
     *       'too few counters', num_events)
      end if

      if (tests_quiet .EQ. 0) then
        print *, 'Number of hardware counters supported: ', num_events
      end if

      call PAPIf_query_event(PAPI_FP_INS, retval)
      if (retval .NE. PAPI_OK) then
        event(1) = PAPI_TOT_INS
      else
C     Total floating point operations
        event(1) = PAPI_FP_INS
      end if

C     Time used
      event(2) = PAPI_TOT_CYC

C     matrix 1: read in the matrix values
      do i=1, nrows1
        do j=1,ncols1
          p(i,j) = i*j*1.0
        end do
      end do

C     matrix 2: read in the matrix values
      do i=1, nrows2
        do j=1,ncols2
          q(i,j) = i*j*1.0
        end do
      end do

C     Initialize the result matrix 
      do i=1,nrows1
        do j=1, ncols2
          r(i,j) = i*j*1.0
        end do
      end do
      
C     Set up the counters
      num_events = 2
      call PAPIf_start_counters( event, num_events, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 
     *       'PAPIf_start_counters', retval)
      end if

C     Clear the counter values
      call PAPIf_read_counters(values, num_events,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 
     *       'PAPIf_read_counters', retval)
      end if

C     Compute the matrix-matrix multiplication
      do i=1,nrows1
        do j=1,ncols2
          do k=1,ncols1
            r(i,j)=r(i,j) + p(i,k)*q(k,j)
          end do
        end do
      end do

C     Stop the counters and put the results in the array values 
      call PAPIf_stop_counters(values,num_events,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 
     *       'PAPIf_stop_counters', retval)
      end if

C     Make sure the compiler does not optimize away the multiplication
      call dummy(r)

      if (tests_quiet .EQ. 0) then

        if (event(1) .EQ. PAPI_TOT_INS) then
          print *, 'TOT Instructions:  ',values(1)
        else
          print *, 'FP Instructions:  ',values(1)
        end if

        print *, 'Cycles: ',values(2)

        if (event(1) .EQ. PAPI_FP_INS) then
          write(*,'(a,f9.6)') ' Efficiency (flops/cycles):',
     &         real(values(1))/real(values(2))
C     Compare measured FLOPS to expected value
          tmp=2.0*real(nrows1)*real(ncols2)*real(ncols1)
          if(abs(values(1)-tmp).gt.tmp*0.05)then
C     Maybe we are counting FMAs?
            tmp=tmp/2.0
            if(abs(values(1)-tmp).gt.tmp*0.05)then
              print *,'Expected operation count:',2.0*tmp
              print *,'Or possibly (using FMA): ',tmp
              print *,'Instead I got:           ',values(1)
              call ftest_fail(__FILE__, __LINE__, 
     *             'Unexpected FLOP count (check vector operations)', 1)
            end if
          end if
        end if
      end if

      call ftests_pass(__FILE__)
      end

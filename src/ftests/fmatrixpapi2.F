C ****************************************************************************
C
C fmatrixpapi2.f
C An example of matrix-matrix multiplication and using PAPI high level to 
C look at the performance. The example illustrates how PAPIF_read_counters
C and PAPIF_accum_counters can be used to selectively measure parts of a
C code without having to use the low-level interface.
C
C Derived from an example written by Kevin London March 2000
C ****************************************************************************

#include "fpapi_test.h"

      program fmatrixpapi
      IMPLICIT integer (p)

      INTEGER ncols1,nrows1,ncols2,nrows2
      PARAMETER(nrows1=175,ncols1=225,nrows2=ncols1,ncols2=150)
      INTEGER i,j,num_events,retval
C   PAPI standardized event to be monitored
      INTEGER event(2)
C   PAPI values of the counters
      INTEGER*8 values(2), dummies(2)
      REAL*8 p(nrows1,ncols1),q(nrows2,ncols2),
     &                 r(nrows1,ncols2)
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()

C   Setup default values
      num_events=0

C   Open matrix file number 1 for reading
C      OPEN(UNIT=1,FILE='fmt1',STATUS='OLD')
C   Open matrix file number 2 for reading
C      OPEN(UNIT=2,FILE='fmt2',STATUS='OLD')

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     .   'PAPI_library_init', retval)
      end if

C   Total floating point operations
        call PAPIf_query_event(PAPI_FP_INS, retval)
        if (retval .NE. PAPI_OK) then
        event(1) = PAPI_TOT_INS
        else
C   Total floating point operations
        event(1) = PAPI_FP_INS
        end if

C  Time used
      event(2) = PAPI_TOT_CYC

C   See how many hardware events at one time are supported
      call PAPIf_num_counters( num_events )
      if ( num_events .LT. 2 ) then
          print *,'This example program requries the architecture ',
     .    'to support 2 simultaneous hardware events...shutting down.'
      stop
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
        call ftest_fail(__FILE__, __LINE__, 
     *'PAPIf_start_counters', retval)
      end if

C  We wish to count the events for this call
      call Adding_MatMult(p,q,r,nrows1,ncols1,ncols2)

C  Read and clear the counter values
      call PAPIf_read_counters(values, num_events,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 
     *'PAPIf_read_counters', retval)
      end if

      if (tests_quiet .EQ. 0) then
        print *
         if (event(1) .EQ. PAPI_TOT_INS) then
          print *, 'TOT Instructions:  ',values(1)
         else
            print *, 'FP Instructions:  ',values(1)
         end if

        print *, 'Cycles: ',values(2)

           if (event(1) .EQ. PAPI_FP_INS) then
             write(*,'(a,f9.6)') ' Efficiency (flops/cycles):',
     &                         real(values(1))/real(values(2))
           end if
      end if

C  We don't wish to count the events for this call
      call Adding_MatMult(p,q,r,nrows1,ncols1,ncols2)

C  Clear the counter values
      call PAPIf_read_counters(dummies, num_events,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 
     *'PAPIf_read_counters', retval)
      end if

C  We wish to count the events for this call
      call Adding_MatMult(p,q,r,nrows1,ncols1,ncols2)

C  Read the counter values
      call PAPIf_accum_counters(values, num_events,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 
     *'PAPIf_accum_counters', retval)
      end if

C  Stop the counters and put the results in the array values 
      call PAPIf_stop_counters(dummies,num_events,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 
     *'PAPIf_stop_counters', retval)
      end if

      if (tests_quiet .EQ. 0) then
        print *
          if (event(1) .EQ. PAPI_TOT_INS) then
                print *, 'TOT Instructions:  ',values(1)
          else
                print *, 'FP Instructions:  ',values(1)
          end if

        print *, 'Cycles: ',values(2)

          if (event(1) .EQ. PAPI_FP_INS) then
               write(*,'(a,f9.6)') ' Efficiency (flops/cycles):',
     &                         real(values(1))/real(values(2))
          end if

        print *
        print *,'----------------------------------------------------'
        print *,'The second instruction and cycle counts should be'
        print *,'approximately twice the first ones. The efficiency'
        print *,'metric should be fairly equal between the cases.'
      end if

      call ftests_pass(__FILE__)
      end

      subroutine Adding_MatMult(p,q,r,ni,nk,nj)
      implicit integer (p)
      integer ni,nk,nj
      real*8 p(ni,*),q(nk,*),r(ni,nj)

      integer i,j,k
C  Compute the matrix-matrix multiplication
      do i=1,ni
       do j=1,nj
         do k=1,nk
           r(i,j)=r(i,j) + p(i,k)*q(k,j)
         end do
       end do
      end do

C  Make sure the compiler does not optimize away the multiplication
      call dummy(r)

      end

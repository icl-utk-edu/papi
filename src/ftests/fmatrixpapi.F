C ****************************************************************************
C
C fmatrixpapi.f
C An example of matrix-matrix multiplication and using PAPI high level to 
C look at the performance. written by Kevin London
C March 2000
C ****************************************************************************

#include "fpapi.h"

      program fmatrixpapi
      INTEGER i, j, k, numevents, ncols1,ncols2,nrows1,nrows2,check
      INTEGER errorcode,length,handle
      CHARACTER*128 errorstring
C   PAPI standardized event to be monitored
      INTEGER event(3)
C   PAPI values of the counters
      INTEGER*8 values(3)
      DOUBLE PRECISION p(175,225),q(225,150),r(175,150)

C   Setup default values
      numevents=0
      ncols1=175
      nrows1=225
      ncols2=225
      nrows2=150
      length = 100
      handle=0

      call PAPIf_library_init( check )
      if ( check.NE.PAPI_VER_CURRENT) then
         print *, 'Error in PAPI_library_init '
         stop
      end if
      call PAPIf_thread_init( handle, handle, check )
      if ( check.NE.PAPI_OK ) then
        print *, 'Error in PAPI_thread_init'
        stop
      end if


C   Open matrix file number 1 for reading
      OPEN(UNIT=1,FILE='fmt1',STATUS='OLD')
C   Open matrix file number 2 for reading
      OPEN(UNIT=2,FILE='fmt2',STATUS='OLD')

C   Total floating point operations
      event(1) = PAPI_FLOPS
C   Level 1 cache misses
      event(2) = PAPI_L1_TCM
C   Level 2 cache misses
      event(3) = PAPI_L2_TCM
      num_events = 3

C   See how many hardware events at one time are supported
      call PAPIf_num_counters( numevents )
      if ( numevents .LT. 3 ) then
          print *, 'Can only monitor ', numevents, ' events.'
          num_events = numevents
      end if
      print *, 'number of hardware counters supported: ', numevents

C   matrix 1: read in the matrix values
      do j=1,ncols1
        do i=1, nrows1
          READ (1,*) p(j,i)
        end do
      end do

C   matrix 2: read in the matrix values
      do j=1,ncols2
        do i=1, nrows2
          READ (2,*) q(j,i)
        end do
      end do

C  Initialize the result matrix 
      do i=1,nrows1
        do j=1, ncols2
	   r(i,j)=0
        end do
      end do

C  Set up the counters
      call PAPIf_start_counters( event, num_events, check)
      if  ( check .LT. 0 ) then
        call PAPIf_perror( check, errorstring, length, errorcode)
        print *, errorstring
        stop
      end if

C  Clear the counter values
      call PAPIf_read_counters(values, num_events,check)
      if ( check .LT. 0) then
        call PAPIf_perror( check, errorstring, length, errorcode)
        print *, errorstring
        stop
      end if

C  Compute the matrix-matrix multiplication
      do i=1,ncols1 
       do j=1,nrows2 
         do k=1,nrows1 
	   r(i,j)=r(i,j) + p(i,k)*q(k,j)
         end do
       end do
      end do

C  Stop the counters and put the results in the array values 
      call PAPIf_stop_counters(values,num_events,check)
      if ( check .LT. 0 ) then
        call PAPIf_perror( check, errorstring, length, errorcode)
        print *, errorstring
        stop
      end if

      print *, 'FLOPS: ', values(1)
      if ( num_events .GT. 1 ) then
        print *, 'Total Level 1 Data cache misses: ', values(2)
      endif
      if ( num_events .GT. 2 ) then
         print *, 'Total Level 2 Data cache misses: ', values(3)
      endif
      return
      end

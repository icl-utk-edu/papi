C ****************************************************************************
C
C matrixpapi.f
C An example of matrix-matrix multiplication and using PAPI low level to 
C look at the performance. written by Kevin London
C March 2000
C ****************************************************************************

#include "fpapi.h"

      program fmatrixlowpapi
      INTEGER i, j, k, ncols1,ncols2,nrows1,nrows2,check
      INTEGER EventSet, length, handle
      CHARACTER*128  errorstring 
      CHARACTER*40 vstring, mstring
C   PAPI standardized event to be monitored
      INTEGER event(3)
C   PAPI values of the counters
      INTEGER*8 values(3)
      DOUBLE PRECISION p(175,225),q(225,150),r(175,150)
      INTEGER*8 starttime,stoptime
      DOUBLE PRECISION finaltime
      INTEGER ncpu,nnodes,totalcpus,vendor,model
      REAL revision, mhz

      numevents=0
      ncols1=175
      nrows1=225
      ncols2=225
      nrows2=150
      EventSet=-1
      starttime=0
      stoptime=0
      length=100
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

C   See how many hardware events at one time are supported
      call PAPIf_num_counters( numevents )

      if ( numevents .LT. 3 ) then
          print *,'This example program requries the architecture ',
     .    'to support 3 simultanious hardware events...shutting down.'
      stop
      end if

      print *, 'number of hardware counters supported: ', numevents
C   Open matrix file number 1 for reading
      OPEN(UNIT=1,FILE='fmt1',STATUS='OLD')
C   Open matrix file number 2 for reading
      OPEN(UNIT=2,FILE='fmt2',STATUS='OLD')

C   Total floating point operations
      call PAPIf_add_event(EventSet,PAPI_FLOPS,check)
      if ( check .LT. 0 ) then
        call PAPIf_perror( check, errorstring, length, errorcode)
        print *, errorstring
        stop
      endif 

C   Level 1 cache misses
      call PAPIf_add_event(EventSet,PAPI_L1_TCM,check)
      if ( check .LT. 0 ) then
        call PAPIf_perror( check, errorstring, length, errorcode)
        print *, errorstring
        stop
      endif 

C   Level 2 cache misses
      call PAPIf_add_event(EventSet,PAPI_L2_TCM,check)
      if ( check .LT. 0 ) then
        call PAPIf_perror( check, errorstring, length, errorcode)
        print *, errorstring
        stop
      endif

C   Grab the hardware info
      call PAPIf_get_hardware_info( ncpu, nnodes, totalcpus, vendor,
     .   vstring, model, mstring, revision, mhz )
      print *, 'A', totalcpus, ' CPU ', mstring, ' at', mhz, 'Mhz.'
      print *, ncpu, nnodes, totalcpus, vendor, vstring, model, 
     .   mstring, revision, mhz

C   matrix 1: read in the matrix values
      do j=1,ncols1
        do i=1, nrows1
          READ (1,*) p(j,i)
        end do
      end do

C   matrix 2: read in the matrix values
      do j=1,nrcols2
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

C  Grab the beginning time
      call PAPIf_get_real_usec( starttime )
 
C  Start the event counters
      call PAPIf_start( EventSet, check )
      if ( EventSet .LT. 0 ) then
        print *, 'Error starting counters'
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
      call PAPIf_stop(EventSet,values,check)
      if ( check .LT. 0 ) then
        print *, 'Error stopping counters'
        stop
      end if

      call PAPIf_get_real_usec( stoptime )
      finaltime = (stoptime/1000000.0) - (starttime/1000000.0)

      print *, 'Time: ', finaltime
      print *, 'FLOPS: ', values(1)
      print *, 'Total Level 1 Data cache misses: ', values(2)
      print *, 'Total Level 2 Data cache misses: ', values(3)
      return
      end

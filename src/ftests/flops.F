C A simple example for the use of PAPI, the number of flops you should
C get is about INDEX^3  on machines that consider add and multiply one flop
C such as SGI, and 2*(INDEX^3) that don't consider it 1 flop such as INTEL
C -Kevin London

#include "fpapi.h"

      program flops

      PARAMETER(index=100)
      REAL*4 matrixa(index,index),matrixb(index,index),mres(index,index)
      REAL*4 proc_time, mflops, real_time
      INTEGER*8 flpins
      INTEGER i,j,k, retval
      integer tests_quiet

      print *, 'Run in quiet mode (enter 1 for yes, 0 for no)?'
      read(5,*) tests_quiet

C Initialize the Matrix arrays
      do i=1,index
        do j=1,index
	   matrixa(i,j) = i+j
	   matrixb(i,j) = j-i
	   mres(i,j) = 0.0
        end do
      end do

C Setup PAPI library and begin collecting data from the counters
      call PAPIf_flops( real_time, proc_time, flpins, mflops, retval )
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error starting PAPI_flops ')
        call ftests_fail(__FILE__,retval)
      end if
C Matrix-Matrix Multiply
      do i=1,index
        do j=1,index
	  do k=1,index
            mres(i,j) = mres(i,j) + matrixa(i,k)*matrixb(k,j)
          end do
        end do
      end do

C Collect the data into the Variables passed in
      call PAPIf_flops( real_time, proc_time, flpins, mflops, retval)
      if ( retval.NE.PAPI_OK) then
        call ftests_perror(__LINE__,'Error stopping PAPI_flops ')
        call ftests_fail(__FILE__,retval)
      end if
      if (tests_quiet .EQ. 0) then
      print *, 'Real_time: ', real_time, ' Proc_time: ', proc_time,
     & ' Total flpins: ', flpins, ' MFLOPS: ', mflops
      end if
      call dummy(mres)

      call ftests_pass(__FILE__)
      end

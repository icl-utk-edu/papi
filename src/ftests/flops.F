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
      INTEGER i,j,k, check

C Initialize the Matrix arrays
      do i=1,index
        do j=1,index
	   matrixa(i,j) = i+j
	   matrixb(i,j) = j-i
	   mres(i,j) = 0.0
        end do
      end do

C Setup PAPI library and begin collecting data from the counters
      call PAPIf_flops( real_time, proc_time, flpins, mflops, check )
      if ( check .LT. PAPI_OK ) then
	print *, 'Error starting PAPI ', check, PAPI_OK
	stop
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
      call PAPIf_flops( real_time, proc_time, flpins, mflops, check )
      print *, 'Real_time: ', real_time, ' Proc_time: ', proc_time,
     & ' Total flpins: ', flpins, ' MFLOPS: ', mflops

      stop
      end

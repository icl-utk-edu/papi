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
      character*25 chbuf

      tests_quiet=0
      call getarg(1,chbuf)
      if (LGE(chbuf, 'TESTS_QUIET')) then
         tests_quiet=1
      end if


#ifdef NO_FLOPS
      call ftest_skip(__FILE__, __LINE__, 'NO_FLOPS check', PAPI_ENOEVNT)
#endif

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
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_flops', retval)
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
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_flops', retval)
      end if
      if (tests_quiet .EQ. 0) then
      print *, 'Real_time: ', real_time
      print *, ' Proc_time: ', proc_time
      print *, ' Total flpins: ', flpins
      print *, ' MFLOPS: ', mflops
      end if
      call dummy(mres)

      call ftests_pass(__FILE__)
      end

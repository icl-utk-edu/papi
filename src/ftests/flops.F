C A simple example for the use of PAPI, the number of flops you should
C get is about INDEX^3  on machines that consider add and multiply one flop
C such as SGI, and 2*(INDEX^3) that don't consider it 1 flop such as INTEL
C -Kevin London

#include "fpapi_test.h"

      program flops
      implicit integer (p)
      integer index

      PARAMETER(index=100)
      REAL*4 matrixa(index,index),matrixb(index,index),mres(index,index)
      REAL*4 proc_time, mflops, real_time
      INTEGER*8 flpins
      INTEGER i,j,k, retval
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()


      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_library_init', retval)
      end if

      call PAPIf_query_event(PAPI_FP_INS, retval)
       if (retval .NE. PAPI_OK) then
        call ftest_skip(__FILE__, __LINE__, 'PAPI_FP_INS', PAPI_ENOEVNT)
       end if

C Initialize the Matrix arrays
      do i=1,index
        do j=1,index
          matrixa(i,j) = i+j
           matrixb(i,j) = j-i
           mres(i,j) = 0.0
        end do
      end do

C Setup PAPI library and begin collecting data from the counters
      call PAPIf_flips( real_time, proc_time, flpins, mflops, retval )
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_flips', retval)
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
      call PAPIf_flips( real_time, proc_time, flpins, mflops, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__, 'PAPIf_flips', retval)
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

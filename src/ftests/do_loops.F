C Compile me with -O0 or else you'll get none. 

#include "fpapi_test.h"

      subroutine do_reads(n)
      implicit none      
      integer n
      integer buf(100)
c L1_MISS_BUFFER_SIZE_INTS)      
      integer i,u,stat
      save u

#ifndef _CRAYT3E
c  Open the data file
      if (u .eq. 0) then
         open (UNIT=1,FILE='/dev/zero', STATUS='OLD', IOSTAT=stat)
         if (stat .ne. 0) then
            write(*,*) '/dev/zero cannot be opened.'
            stop
         end if
	 u = 1
      end if

c  Read the number of points
      do i = 1, n
        read(UNIT=1,FMT=*,IOSTAT=stat) buf
        if (stat .ne. 0) then
            write(*,*) '/dev/zero cannot be read.'
            stop
         end if
      end do

c  Close the file
      close (u)
#endif
      end

      subroutine do_flops(n)
      implicit none      
      integer n
      real*8 a,b,c
      common a, b
      integer i
      c = 0.11
      do i = 1, n
        c = c + a * b
      end do
#ifndef _CRAYT3E
      call dummy(c)
#endif
      end

      subroutine do_l1misses(n)
      implicit none      
      integer n
      integer i, j
      integer buf(L1_MISS_BUFFER_SIZE_INTS)      
      do i = 1, n
        do j=1, L1_MISS_BUFFER_SIZE_INTS
          buf(j)=buf(L1_MISS_BUFFER_SIZE_INTS-j)+1
        end do
      end do
      end

      subroutine do_both(n)
      implicit none      
      integer n
      integer i, flops, reads
      flops = NUM_FLOPS / n
      reads = NUM_READS / n
      do i = 1, n
         call do_flops(flops)
         call do_reads(reads)
      end do
      end

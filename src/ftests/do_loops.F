C Compile me with -O0 or else you'll get none. 

#include "fpapi_test.h"

      subroutine do_reads(n)
      implicit none      
      integer n
      integer buf(L1_MISS_BUFFER_SIZE_INTS)      
      integer i,u,stat
      save u

c  Open the file
      if (u .eq. 0) then
         open (UNIT=1,STATUS='SCRATCH', IOSTAT=stat)
         if (stat .ne. 0) then
            write(*,*) 'Temporary file cannot be opened.'
            stop
         end if
	 u = 1

c  Fill the file
      do i = 1, n
        write(u,*,IOSTAT=stat) buf
        if (stat .ne. 0) then
            write(*,*) 'Temporary file cannot be written.'
            stop
        end if
      end do
      rewind(u)
      end if

c  Read the file
      do i = 1, n
        read(u,*,IOSTAT=stat) buf
        if (stat .ne. 0) then
            write(*,*) 'Temporary file cannot be read.'
            stop
        end if
      end do
      
c  Rewind the file as this function may be called more than once
      rewind(u)
      end

      subroutine do_flops(n)
      implicit none      
      integer n
      real*8 a,b,c
c     common a, b
      integer i
      a = 0.5
      b = 6.2
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

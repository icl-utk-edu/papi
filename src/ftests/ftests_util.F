#include "fpapi.h"
#include "tests.h"

#define L1_MISS_BUFFER_SIZE_INTS 128*1024

      subroutine dummy(array)
      implicit none
      real array
      
C    This is a dummy routine that can be used to make the
C    compiler believe that the result of a loop is actually
C    used in the program. See eg. fmatrixpapi.F

      end

      integer function len_trim(string)
      implicit none
      character*(*) string

      do len_trim=len(string),1,-1
        if(string(len_trim:len_trim).NE.' ') return
      end do
      len_trim=0
      end

      subroutine ftests_warning(line,msg)
      implicit none
      character*(*) msg
      integer line

      write(*,*) '****  WARNING message  ****'
      call ftests_perror(line,msg)
      end

      subroutine ftests_fatal_error(line,msg)
      implicit none
      character*(*) msg
      integer line

      write(*,*) '****  ERROR message  ****'
      call ftests_perror(line,msg)
      write(*,*) '****  Test aborts  ****'
      stop
      end

      subroutine ftests_perror(line,msg)
      implicit none
      character*(*) msg
      integer line

      write(*,*) '****  Test error occurred  ****'
      write(*,100) line,msg
  100 format(t3,'At source line',i5,':: ',a)

      end
      
      subroutine ftests_pass(test_str)
      implicit none
      character*(*) test_str

      write(*,100) test_str,':                PASSED'
      call PAPIF_shutdown()
      stop
 100  format(/,t1,a,a)
      end

      subroutine ftests_fail(test_str,papiretval)
      implicit none
      character*(*) test_str
      character*(PAPI_MAX_STR_LEN) papi_errstr
      integer papiretval
      integer len_trim
      external len_trim

      write(*,*) test_str,':                FAILED'
      if(papiretval.lt.0)then
        call PAPIF_perror(papiretval,papi_errstr,len(papi_errstr))
        write(*,*) 'PAPI error: ',papi_errstr(1:len_trim(papi_errstr))
      end if
      stop
      end


C
C  TEST UTILITIES
C
      subroutine do_flops(n)
      integer n
      implicit none      
      double precision a, b, c
      integer i
      a = 0.5
      b = 6.2
      c = 0.11
      do i = 1, n
         c = c + a * b
      end do
      end

      subroutine do_l1misses(n)
      integer n
      implicit none      
      integer i, j
      integer buf(L1_MISS_BUFFER_SIZE_INTS)      
      do i = 1, n
         do j=1, L1_MISS_BUFFER_SIZE_INTS
            buf(j)=buf(L1_MISS_BUFFER_SIZE_INTS-j)+1
         end do
      end do
      end

      subroutine do_both(n)
      integer n
      implicit none      
      double precision a, b, c
      integer i, j
      integer buf(L1_MISS_BUFFER_SIZE_INTS)      
      a = 0.5
      b = 6.2
      c = 0.11
      do i = 1, n
         do j=1, n
            c = c + a * b
         end do
         do j=1, L1_MISS_BUFFER_SIZE_INTS
            buf(j)=buf(L1_MISS_BUFFER_SIZE_INTS-j)+1
         end do
      end do
      end

      subroutine stringify_domain(domain, str)
      integer domain
      character*(PAPI_MAX_STR_LEN) str

      if (domain .EQ. PAPI_DOM_USER) then
         str = "PAPI_DOM_USER"
      else if (domain .EQ. PAPI_DOM_KERNEL) then
         str = "PAPI_DOM_KERNEL"
      else if (domain .EQ. PAPI_DOM_OTHER) then
         str = "PAPI_DOM_OTHER"
      else if (domain .EQ. PAPI_DOM_ALL) then
         str = "PAPI_DOM_ALL"
      else
         print *, 'error in stringify_domain'
         stop
      end if
      end

      subroutine stringify_granularity(granularity, str)
      integer granularity
      character*(PAPI_MAX_STR_LEN) str

      if (granularity .EQ. PAPI_GRN_THR) then
         str = "PAPI_GRN_THR"
      else if (granularity .EQ. PAPI_GRN_PROC) then
         str = "PAPI_GRN_PROC"
      else if (granularity .EQ. PAPI_GRN_PROCG) then
         str = "PAPI_GRN_PROCG"
      else if (granularity .EQ. PAPI_GRN_SYS_CPU) then
         str = "PAPI_GRN_SYS_CPU"
      else if (granularity .EQ. PAPI_GRN_SYS) then
         str = "PAPI_GRN_SYS"
      else
         print *, 'error in stringify_granularity'
         stop
      end if
      end


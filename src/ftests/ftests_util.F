#include "fpapi.h"

#define L1_MISS_BUFFER_SIZE_INTS 128*1024

      integer function get_quiet()
#if _WIN32
      use dflib
#endif
      implicit none
      character*25 chbuf
	integer quiet
	common quiet

C	This routine tests for a command line argument
C	that matches 'TESTS_QUIET'
C	If found, it returns 1 to set output to quiet
C	The routine was placed her to hide the ugly
C	Windows #if stuff from normal view
C	And also to make the test code read cleaner
      get_quiet=0
      call getarg(1,chbuf)
      if (LGE(chbuf, 'TESTS_QUIET')) then
         get_quiet=1
      end if
	quiet=get_quiet
	end


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

      call ftests_perror(line,msg)
	call pause()
      stop
      end

      subroutine ftests_perror(line,msg)
      implicit none
      character*(*) msg
      integer line

      write(*,*) '****  Test error occurred  ****'
      write(*,100) line,msg
  100 format(t3,'Line # ',i5,':: ',a)

      end
      
      subroutine ftests_pass(test_str)
      implicit none
      character*(*) test_str

      write(*,*) test_str,':                PASSED'
      call PAPIF_shutdown()
	call pause()
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

      if ( papiretval.eq.PAPI_ESBSTR ) then
        call ftests_skip(test_str,papiretval)
      end if
      if ( papiretval.eq.PAPI_ENOEVNT ) then
        call ftests_skip(test_str,papiretval)
      end if

      if ( papiretval.ne.0 )then
          write(*,*) test_str,':                FAILED'
      else
          write(*,*) test_str,':                SKIPPED'
      end if
      if(papiretval.lt.0)then
        call PAPIF_perror(papiretval,papi_errstr,len(papi_errstr))
        write(*,*) 'PAPI error: ',papi_errstr(1:len_trim(papi_errstr))
      end if
      if(papiretval.eq.0)then
        write(*,*) 'SGI requires root permissions for this test\n'
      end if
	call pause()
      stop
      end

      subroutine ftests_skip(test_str,papiretval)
      implicit none
      character*(*) test_str
      character*(PAPI_MAX_STR_LEN) papi_errstr
      integer papiretval
      integer len_trim
      external len_trim

      write(*,*) test_str,':               SKIPPED' 
      if(papiretval.lt.0)then
        call PAPIF_perror(papiretval,papi_errstr,len(papi_errstr))
        write(*,*) 'PAPI error: ',papi_errstr(1:len_trim(papi_errstr))
      end if
	call pause()
      stop
      end

      subroutine ftest_fail(file, line, callstr, retval)
      implicit none
      character*(*) file
      integer line
      character*(*) callstr
      integer retval
      character*(PAPI_MAX_STR_LEN) papi_errstr
      integer len_trim
      external len_trim

      if ( retval.eq.PAPI_ESBSTR ) then
        call ftest_skip(file, line, callstr, retval)
      end if
      if ( retval.eq.PAPI_ENOEVNT ) then
        call ftest_skip(file, line, callstr, retval)
      end if

      if ( retval.ne.0 )then
          write(*,*) file, ':                FAILED'
          write(*,*) 'Line # ', line
      else
          write(*,*) file, ':                SKIPPED'
          write(*,*) 'Line #', line
      end if

      if (retval.eq.PAPI_ESYS) then
         write(*,*) "System error in ", callstr
      else if (retval.gt.0) then
         write(*,*) "Error calculating: ", callstr
      else if(retval.eq.0)then
        write(*,*) 'SGI requires root permissions for this test'
      else
        call PAPIF_perror(retval,papi_errstr,len(papi_errstr))
        write(*,*) 'PAPI error in ', callstr, 
     *': ', papi_errstr(1:len_trim(papi_errstr))
      end if
	call pause()
      stop
      end

      subroutine ftest_skip(file, line, callstr, retval)
      implicit none
      character*(*) file
      integer line
      character*(*) callstr
      integer retval

      character*(PAPI_MAX_STR_LEN) papi_errstr
      integer len_trim
      external len_trim

      write(*,*) file, ': SKIPPED'
      write(*,*) 'Line #', line
      if(retval.eq.PAPI_ESYS) then
         write(*,*) "System error in ", callstr
      else if (retval.gt.0) then
         write(*,*) "Error calculating: ", callstr
      else
        call PAPIF_perror(retval,papi_errstr,len(papi_errstr))
        write(*,*) 'Error in ', callstr, ' : ', 
     *       papi_errstr(1:len_trim(papi_errstr))
      end if
	call pause()
      stop
      end


C
C  TEST UTILITIES
C
      subroutine do_flops(n)
      implicit none      
      integer n
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
      implicit none      
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
	   call pause()
         stop
      end if
      end

      subroutine stringify_granularity(granularity, str)
      implicit none      
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
	   call pause()
         stop
      end if
      end

C	This routine provides a bottleneck
C	at the exit point of a program
C	For Windows, it links to a simple C routine
C	that prompts the user for a keypress
C	For every other platform it is a nop.
      subroutine pause()
      implicit none      
	integer quiet,i
	common quiet

#ifdef _WIN32
	if (quiet .eq. 0) then
	  call cpause()
	end if
#endif
	end

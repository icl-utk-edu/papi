#include "fpapi.h"

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
      stop ''
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

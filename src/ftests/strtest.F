C  Strtest - Perform some basic tests of the functionality of the
C  string passing to and from the PAPI Fortran interface. 
C
C    Test 1: Look up an event name from an event code. Use this name
C    to try and locate the event code using the name received. Long,
C    short and too short strings are used in the tests
C
C    Test 2: Look up a PAPI error string. Use long, short and too 
C    short strings to store the result.
C
C    Comments:
C    When using the Fortran interface it may not always be possible to
C    use the PAPI predefined constants as actual arguments. Due to the
C    values in these compilers might occasionally cast these into the
C    wrong type. In the code below the line code=MSGCODE is used to make
C    sure that the event code get the right type.
C
#include "fpapi.h"
C Set MSGLEN to the number of characters in the named event in MSGCODE
#define MSGLEN 11
#define MSGCODE PAPI_L1_DCM 
#define ERRCODE PAPI_EINVAL
      
      program strtest
      implicit none
      
      CHARACTER*(PAPI_MAX_STR_LEN) papistr,papierr
      CHARACTER*(PAPI_MAX_STR_LEN/2) papihalfstr
      CHARACTER*(PAPI_MAX_STR_LEN*2) papidblstr

      CHARACTER*(MSGLEN) invstr1
      CHARACTER*(MSGLEN+1) invstr2
      CHARACTER*(MSGLEN+2) invstr3
      CHARACTER*(MSGLEN-1) invstr4
      CHARACTER*(MSGLEN-2) invstr5

      integer i,check,len,code,code2,retcode,lastchar,papicode
      integer getstrlen
      external getstrlen

      check=PAPI_VER_CURRENT
      call PAPIF_library_init(check)
      if(check.ne.PAPI_VER_CURRENT)then
        call PAPIF_perror(check,papierr,code)
        print *,papierr
        stop 'Library init error.'
      end if

      print *,'---------------------------------------------------'
      print *,'   Testing PAPIF_name_to_code/PAPIF_code_to_name   '
      print *,'---------------------------------------------------'
      print *,' These tests look up an event name and event code'
      print *,' On no occasion a NULL character should be found(+)'
      print *,' When too short strings are used, the lookup should fail'
      print *
      code=MSGCODE
      print *,' Tests use the event code ',code
      print *

 100  format(t1,a,i4)
      print 100,"Testing string length ",len(papistr)
      call checkstr(code,papistr,retcode,check)
      print *
      lastchar=getstrlen(papistr)
      
      print 100,"Testing string length ",len(invstr1)
      if(len(invstr1).lt.lastchar)then
        print *,'This call should return an error code.'
      end if
      call checkstr(code,invstr1,retcode,check)
      print *

      print 100,"Testing string length ",len(invstr2)
      if(len(invstr2).lt.lastchar)then
        print *,'This call should return an error code.'
      end if
      call checkstr(code,invstr2,retcode,check)
      print *

      print 100,"Testing string length ",len(invstr3)
      if(len(invstr3).lt.lastchar)then
        print *,'This call should return an error code.'
      end if
      call checkstr(code,invstr3,retcode,check)
      print *

      print 100,"Testing string length ",len(invstr4)
      if(len(invstr4).lt.lastchar)then
        print *,'This call should return an error code.'
      end if
      call checkstr(code,invstr4,retcode,check)
      print *

      print 100,"Testing string length ",len(invstr5)
      if(len(invstr5).lt.lastchar)then
        print *,'This call should return an error code.'
      end if
      call checkstr(code,invstr5,retcode,check)
      print *

      print *,'---------------------------------------------------'
      print *,'          Testing PAPIF_perror                     '
      print *,'---------------------------------------------------'
      print 100,' These tests should return a PAPI error string for'//
     $    ' error',ERRCODE
      print *,' On no occasion a NULL character should be found(+)'
      print *

      call test_papif_perror(ERRCODE,papistr,check)
      print *

      call test_papif_perror(ERRCODE,papihalfstr,check)
      print *

      call test_papif_perror(ERRCODE,papidblstr,check)
      print *

      call test_papif_perror(ERRCODE,invstr1,check)
      print *


      print *,'---------------------------------------------------'
      print *,'          Testing PAPIF_descr_event                '
      print *,'---------------------------------------------------'
      print *,' These tests should return a PAPI description for'
      print *,' various event names and argument shapes.'
      print *,' On no occasion a NULL character should be found(+)'
      print *

      print *,'Test 1'
      papistr="PAPI_L1_DCM"
      papicode=-1
      call test_papif_descr(papistr,papicode,papidblstr,check)
      if(papicode.ne.PAPI_L1_DCM)then
        print *,'Code look up failed?'
      end if
      print *

      print *,'Test 2'
      papistr="  "
      papicode=PAPI_L2_DCM
      call test_papif_descr(papistr,papicode,papidblstr,check)
      i=getstrlen(papistr)
      if(papistr(1:i).ne."PAPI_L2_DCM")then
        print *,'Name look up failed?'
        print *,'Got:      "'//papistr(1:i)//'"'
        print *,'Expected: "'//'PAPI_L2_DCM'//'"'
      end if
      print *

      print *,'Test 3'
      invstr1="PAPI_L1_ICM"
      papicode=-1
      call test_papif_descr(invstr1,papicode,papidblstr,check)
      if(papicode.ne.PAPI_L1_ICM)then
        print *,'Code look up failed?'
      end if
      print *

      print *,'Test 4'
      invstr1="  "
      papicode=PAPI_L2_ICM
      call test_papif_descr(invstr1,papicode,papidblstr,check)
      i=getstrlen(invstr1)
      if(invstr1(1:i).ne."PAPI_L2_ICM")then
        print *,'Name look up failed?'
        print *,'Got:      "'//invstr1(1:i)//'"'
        print *,'Expected: "'//'PAPI_L2_ICM'//'"'
      end if
      print *

      print *,'Test 5  (This test should get a truncated description)'
      invstr2="PAPI_L3_DCM"
      papicode=-1
      call test_papif_descr(invstr2,papicode,invstr1,check)
      if(papicode.ne.PAPI_L3_DCM)then
        print *,'Code look up failed?'
      end if
      print *

      print *,'Test 6  (This test should get a truncated description)'
      invstr2="  "
      papicode=PAPI_L3_ICM
      call test_papif_descr(invstr2,papicode,invstr1,check)
      i=getstrlen(invstr2)
      if(invstr2(1:i).ne."PAPI_L3_ICM")then
        print *,'Name look up failed?'
        print *,'Got:      "'//invstr2(1:i)//'"'
        print *,'Expected: "'//'PAPI_L2_ICM'//'"'
      end if
      print *

      print *,'Test 7  (This test should get a truncated name)'
      invstr4="  "
      papicode=PAPI_L1_DCM
      call test_papif_descr(invstr4,papicode,papistr,check)
      i=getstrlen(invstr4)
      if(invstr4(1:i).ne."PAPI_L1_DCM")then
        print *,'Name look up failed?'
        print *,'Got:      "'//invstr4(1:i)//'"'
        print *,'Expected: "'//'PAPI_L1_DCM'//'"'
      end if

      print *,'---------------------------------------------------'
      print *,'(+) Fortran implementations that do not provide the'
      print *,'    string argument length might show NULL characters.'
      print *,'    This may or may not be OK depending on the Fortran'
      print *,'    compiler. See papi_fwrappers.c and your Fortran'
      print *,'    compiler reference manual.'
      end

      subroutine checkstr(incode,string,outcode,check)
      implicit none
      integer incode,outcode,check
      character*(*) string
      character*(PAPI_MAX_STR_LEN) papierr
      integer code,i
      integer getstrlen
      external getstrlen

      code=incode
      call PAPIF_event_code_to_name(code,string,check)
      if(check.ne.PAPI_OK)then
        call PAPIF_perror(check,papierr,code)
        print *,'*ERROR* '//papierr(1:getstrlen(papierr))
        print *,'******* '//'Error in checkstr using '//
     $      'PAPIF_event_code_to_name'
      end if
      i=index(string,char(0))
      if(i.gt.0)then
        print *,'NULL character found in string!!!'
      end if

      print *,'The event name is: "'//string(1:getstrlen(string))//'"'
      
      call PAPIF_event_name_to_code(string,code,check)
      if(check.ne.PAPI_OK)then
        call PAPIF_perror(check,papierr,code)
        print *,'*ERROR* '//papierr(1:getstrlen(papierr))
        print *,'******* '//'Error in checkstr using '//
     $      'PAPIF_event_name_to_code'
      end if
      
      return
      end

      subroutine test_papif_perror(incode,string,check)
      implicit none
      integer incode,check
      character*(*) string
      integer code,i
      integer getstrlen
      external getstrlen

      code=incode
      call PAPIF_perror(code,string,check)
      print 100,'Testing string of length ',len(string)
      i=getstrlen(string)
      print *,'"1234567890....5...20....5...30....5...40....5...50...'
      print *,'"'//string(1:i)//'"'
      i=index(string,char(0))
      if(i.gt.0)then
        print *,'NULL character found in string!!!'
      end if
      
 100  format(t1,a,i4)
      end

      subroutine test_papif_descr(name,code,string,check)
      implicit none
      integer code,check
      character*(*) name,string

      character*(PAPI_MAX_STR_LEN) papierr
      integer i
      integer getstrlen
      external getstrlen

      call PAPIF_describe_event(name,code,string,check)
      print *,'The event description is: "'//
     $    string(1:getstrlen(string))//'"'
      
      if(check.ne.PAPI_OK)then
        call PAPIF_perror(check,papierr,code)
        print *,'*ERROR* '//papierr(1:getstrlen(papierr))
        print *,'******* '//'Error in test_papif_descr using '//
     $      'PAPIF_describe_event'
      end if
      i=index(string,char(0))
      if(i.gt.0)then
        print *,'NULL character found in string!!!'
      end if
      i=index(name,char(0))
      if(i.gt.0)then
        print *,'NULL character found in name!!!'
      end if

      return
      end

      integer function getstrlen(string)
      implicit none
      character*(*) string
      integer i

      do i=len(string),1,-1
        if(string(i:i).ne.' ') then 
          goto 20
        end if
      end do
      getstrlen=0
      return

 20   continue
      getstrlen=i
      return
      end

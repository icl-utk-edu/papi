#include "header.h"
program MAPS
!
! This code was writen by Laura C. Carrington and was
! derived from Stuart Johnsons original
! memory bandwidth tests for SDSCs IBM Blue Horizon.
! This code tests the memory bandwidth of various levels
! of memory heirarchy.  It can also test the effects of 
! memory bound code running on more than one cpu of an SMP
! node.  The main memory bandwidth testing loop is of step
! size 1, but can be changed to step through randomly by
! changing the main variable "step" to be true in the
! RunConditions file.
!
! parameters:
! noff 		is the main x chunk offset and should
!   be a large power of 2 greater than any value
!   of n you plan to use in the inner loop. 
!   So your inner loop tests L1 and L2 Cache to make
!   noff = 2* L2s. Also,make sure that noff >> l2off+l1off.
!   I have set noff to 1Gb/512=2*L2s, which allows n up
!   to 2,000,000 +, which means you can use at least
!   4 times the cache size for the memory bandwidth test.
!
! l2off 	is the offset between the different streams
!   For BH l2off should be a multiple of 64w (4 lines) which is 
!   the size of a congruence class of the 128-way L1 cache.
!   For BH l2off is set to 4 time the size of l1 cache which is 
!   also 1/2 the size of a congruence class in L2 cache.
!   For the T3E l2off being 4*L1 cache = 4*1024w would be the
!   same size as a congruence class on L2 cache.  So making it 
!   2*1024 might be better.
!
! l1off 	This is is cache line length in double 
!   precision words and the offset between congruence
!   classes in L1 cache on BH which is 16 words. 
!   For the T3E a L1 cache line is 4 words and L2 cache line is 
!   8 words.
!
  implicit none
#if MODE==PARALLEL
  include 'mpif.h'
#endif
  integer noff,l2off,l1off,poff
#if MACHINE==IBM_BH
  ! noff=2*L2s, l2off=4*L1s, l1off = line size
  parameter(noff=1024*1024*1024/512,l2off=32768,l1off=16,poff=16)
#elif MACHINE==IBM_LH
  parameter(noff=2*512*1024, l2off=noff + 256*1024 + 11.25*1024 - 256)
  parameter(l1off=0,poff=16)
#elif MACHINE==T3E
!
!  for the T3E l1off=4 for Dcache and l1off=8 for Scache size of L1s=1024*8
!
! noff+l1off is offset for a stream   noff=L2s better than noff=4*1024
!
  parameter(noff=3*4*1024, l2off=4*1024+512, l1off=8, poff=8)
#elif MACHINE==TCS1
!  L1s=1024*8  L2s=1024*1024
!  L2s/2=128*4*1024
!  1/2 each block (2-way s.a) for L1 = 2*1024
!  off set for stream = 128*4*1024 + 2*1024+64
!
  parameter(noff=1024*1024, l2off=128*4*1024+2*1024+8, l1off=8, poff=8)
#elif MACHINE==TICK
!  L1s=1024*4  L2s=512*1024
!  L2s/2=256*1024
!  1/2 each block (2-way s.a) for L1 = 2*1024
!
  parameter(noff=512*1024, l2off=2*512*1024+256*1024+1024, l1off=8, poff=8)
#elif MACHINE==SUNHPC
!
! DO NOT USE L2OFF, L1OFF, POFF
!
  parameter(noff=8*1024*1024, l2off=8, l1off=8, poff=8)
#elif MACHINE==ITANIUM1
!
!  assuming L1=L2 and L2=L3
!  L1s=12*1024 L2s=512*1024
!  congruence class size for L1=2*1024 (6-way)
!  congruence class size for L2=64*1024=32*2*1024 (4-way)
!  streams for L1=1024 (+8)
!  streams for L2=16*2*1024+1024 (+8)
!
  parameter(noff=512*1024, l2off=16*2*1024+1024, l1off=8, poff=8)
#elif MACHINE==MCKINLEY
!!!!MCK
!  noff = 2*L3
   parameter(noff=2*3*128*1024, l2off=8*1024+1024, l1off=8,poff=8)
#endif
  real*8 y(16),x(12*noff),pad(16),pad2(16)
  integer*4 index(12*noff)
  integer L1s,L2s,L3s
#if MACHINE==IBM_BH
  integer L1_line,L2_line,L1_size,L2_size
#endif

#if MODE==PARALLEL
!
!  MPI variables
!    
  integer iam, ierr, npes
#elif MODE==SERIAL
  integer iam
#endif
#if HPM==COUNT
  integer counter, hpmflag
  common /cnt/counter,hpmflag
#endif
#if MACHINE==IBM_BH
  common x
#else
  common /arry/x,pad,y,pad2,index
!DIR$ CACHE_ALIGN /arry/
#endif

#if MODE==PARALLEL
!
! Initialize MPI
!
  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD,iam,ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,npes,ierr)
#elif MODE==SERIAL
  iam=0
#endif

#if HPM==COUNT
  call f_hpminit(0,"non-random load")
  counter=0
#endif
	
  call random_number(x)
  call random_number(y)
!
! get cache line details from c code bw_g.c
!
#if MACHINE==T3E
  L1s=1024 !words
  L2s=12*1024 !words
#elif MACHINE==IBM_BH
  L1s=8*1024
  L2s=1024*1024
#elif MACHINE==IBM_LH
  L1s=4*1024
  L2s=184320
  L3s=4*1024*1024
#elif MACHINE==TCS1
  L1s=8*1024  !words
  L2s=1024*1024  !words
#elif MACHINE==TICK
  L1s=4*1024  !words
  L2s=512*1024 !words
#elif MACHINE==SUNHPC
  L1s=2*1024
  L2s=512*1024
#elif MACHINE==ITANIUM1
  L1s=12*1024  !Itanium
  L2s=512*1024  !Itanium
#elif MACHINE==MCKINLEY
  L1s=32*1024  !MCK
  L2s=3*128*1024 !MCK
#endif
!
! Start memory bandwidth test 
!
  call mem_step(y,x,index,noff,l2off,l1off,poff,&
     &                  L1s,L2s,L3s,iam)
#if HPM==COUNT
  call f_hpmterminate(0)
#endif
#if MODE==PARALLEL
!
! Close file and finalize
!
  call MPI_FINALIZE(ierr)
#endif
  stop
  end
	
!************************************************************
!***************    SUBROUTINE MEM_STEP   *******************
!************************************************************
  subroutine mem_step(y,x,index,noff,l2off,l1off,poff,L1s,L2s,L3s,&
     &				iam)
!
! This routine tests loads and stores for L1,L2, and main Memory
!
  implicit none
#if MODE==PARALLEL
  include 'mpif.h'
#endif
  character(LEN=12) name
  integer nrep,n,noff,l2off,l1off,poff,main,nrep_start
  integer ns,nss,L1s,L2s,L3s,tests,startn,unit
  real*8 y(16),x((4+8)*noff),recommended_time,cpu_MHz
  integer nStart,total,num_streams
  logical test_stores,test_loads,read_from_file,step,test_streams
  external load,store
  integer iam,n_org
  integer*4 index(12*noff)
!
! Open input for run conditions of the test
!
  open(8,file="RunConditions",status="old")
    read(8,*)tests
    read(8,*)test_stores
    read(8,*)test_loads
    read(8,*)test_streams
    read(8,*)num_streams
    read(8,*)step
    read(8,*)nStart
    read(8,*)startn
    read(8,*)total
    read(8,*)read_from_file
    read(8,*)recommended_time
    read(8,*)nrep_start
    read(8,*)cpu_MHz
  close(8)
  nrep=nrep_start	
!
! Testing loads for L1 to MM:
!  n is the number of elements of the array so if L1 cache<n<L2 cache
!  then you are testing L2 cache. The value of n (test size) is 
!  incremented either by value "startn" or the next value read from 
!  the file Iterations.
!
  if(test_loads)then     !If true the test for loads
!
!   open file and write header if task 0
!
    if (iam .eq. 0)then
      unit=15
#if MODE==PARALLEL
      open(unit,file="Load_mpi",status="unknown")
#elif MODE==SERIAL
      open(unit,file="Load",status="unknown")
#endif
      write(unit,*)'Testing Loads on different levels of memory'
      write(unit,200)'Memory Level','L/S','size','# Streams ',&
                &  'step','cycles/load','MB/s'
    end if

    if (read_from_file)then
      !If incrementing size n by value in file then read in first
      ! value of file, which is total number of sizes to test
      open(8,file="Iteration",status="old")
        read(8,*)total
    end if

    do main=1,total
      if (read_from_file) read(8,*)n  !read in test size
#if MACHINE==T3E
      if (n .le. L2s) then
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      if (n .le. L1s) then
#elif MACHINE==TCS1 || MACHINE==SUNHPC || MACHINE==ITANIUM1 || MACHINE==TICK || MACHINE==MCKINLEY
      if (n .lt. 15*L2s) then    !assuming no streams for TCS1 for now
#endif
        n_org=n
        call testing(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
        &  name,L1s,L2s,L3s,'Load ',tests,load,recommended_time,cpu_MHz,unit)
        n = n_org
      else
         if(test_streams)then
           do ns = 1,num_streams
              n_org = n
!             ns = num_streams
             call testing(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
          &  name,L1s,L2s,L3s,'Load ',tests,load,recommended_time,cpu_MHz,unit)
	      n = n_org
	   end do
         else
           ns=1
           n_org = n
           call testing(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
         &  name,L1s,L2s,L3s,'Load ',tests,load,recommended_time,cpu_MHz,unit)
           n = n_org
         endif
      end if
      if (.not. read_from_file)then
        n=n+startn
      end if
    end do
    close(8)
  end if
!
! testing store for L1 to MM
!
  if(test_stores)then  !Testing Stores
!
!   open file and write header if task 0
!
    if (iam .eq. 0)then
      unit=15
#if MODE==PARALLEL
      open(unit,file="Store_mpi",status="unknown")
#elif MODE==SERIAL
      open(unit,file="Store",status="unknown")
#endif
      write(unit,*)'Testing Stores on different levels of memory'
      write(unit,200)'Memory Level','L/S','size','# Streams ',&
                    & 'step','cycles/store','MB/s'
    end if

    if (read_from_file)then
      !If incrementing size n by value in file then read in first
      ! value of file, which is total number of sizes to test
      open(8,file="Iteration",status="old")
        read(8,*)total
    end if

    nrep=nrep_start !reset nrep if did load test above

    ! Main loop to loop through all test sizes
    do main=1,total
      if (read_from_file) read(8,*)n !read in test size
#if MACHINE==T3E
      if (n .le. L2s) then
        ns=1
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      if (n .le. L1s) then
#elif MACHINE==TCS1 || MACHINE==SUNHPC || MACHINE==ITANIUM1 || MACHINE==TICK || MACHINE==MCKINLEY
      if (n .lt. 15*L2s) then    !assuming no streams for TCS1 for now
#endif
        n_org = n
        call testing(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
         & name,L1s,L2s,L3s,'Store',tests,store,recommended_time,cpu_MHz,unit)
        n = n_org
      else
        if (test_streams)then
#if MACHINE==T3E || MACHINE==TCS1 || MACHINE==TICK
          do nss = 1,num_streams
#elif MACHINE==IBM_BH || MACHINE==SUNHPC || MACHINE==ITANIUM1 || MACHINE==MCKINLEY || MACHINE==IBM_LH
          do nss = 0,num_streams
#endif
            ns=nss
            !ns=num_streams
            n_org = n
            call testing(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
            & name,L1s,L2s,L3s,'Store',tests,store,recommended_time,cpu_MHz,unit)
 	    n = n_org
          end do
        else
#if MACHINE==T3E || MACHINE==TCS1 || MACHINE==TICK
          ns=1
#elif MACHINE==IBM_BH || MACHINE==SUNHPC || MACHINE==ITANIUM1 || MACHINE==MCKINLEY
          ns=0
#endif
          n_org = n
          call testing(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
           & name,L1s,L2s,L3s,'Store',tests,store,recommended_time,cpu_MHz,unit)
          n = n_org
        endif
      end if
      if (.not. read_from_file)then
        n=n+startn
      end if
    end do
    close(8)
  end if
200     format(A12,',',A5,',',A10,',',A11,',',A4,',',A12,',',A5)
  if (iam .eq. 0) close(unit)
  return
  end subroutine mem_step


!************************************************************
!***************    SUBROUTINE STORE   **********************
!************************************************************
  subroutine store(nrep,n,y,x,index,nstreams,noff,l2off,l1off,&
                 &         step,iam,cyc,mb,t,cpu_MHz)
!
! This routine measures store bandwidth of the different levels of memory.
! Which level is tested depends on the value of n.  If n<size of L1 cache
! then mainly the bandwidth of L1 cache is tested. If L1 size<n<L2 size then 
! mainly the bandwidth of L2 cache is tested. If L2 size<n<memory then mainly
! main memory is tested. Note that I say mainly because bandwidth test is sensitive
! to the size of n and its size relative to the different levels of memory.
!
  implicit none
#if MODE==PARALLEL
  include 'mpif.h'
  integer ierr
#endif
  integer irep,i,nrep,n,noff,l2off,l1off,poff
  integer nstreams,iam,ns,ntemp
  real*8 y(16),x((4+8)*noff),cyc,mb,cpu_MHz,num
  real*8 time1, time2,second,t,totalt
#if MACHINE==SUNHPC
  integer dummy
#endif
#if MACHINE==IBM_BH
  parameter (poff=16)
#endif
  logical step
  integer*4 index(12*noff)
!
! If the step size is true then call another routine with similar loops 
! to those below only with variable indexing.  This variable indexing 
! determines bandwidth for stride-random stores.
!
#if MACHINE==SUNHPC
  dummy=1024-2**10
#endif
  ns = nstreams
  if (step) then
    call store_random(nrep,n,y,x,index,nstreams,noff,l2off,l1off,&
           &  poff,iam,cyc,mb,t,cpu_MHz)
    return
  end if

!
! zero streams can happen for stores because to get 1 stream
! you must use a prefetch directive.  Testing zero stream store.
!
  if (nstreams .eq. 0) then
    do irep=1,nrep
      do i=1,n
#if MACHINE==SUNHPC
        x(i + dummy*irep)=1.0d0
#else
        x(i)=1.0d0
#endif
      end do
    end do
!
! use barrier to make sure all PEs are hitting the loop at the 
! same time.
!
#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif
    time1 = second()
    do irep=1,nrep
      do i=1,n
#if MACHINE==SUNHPC
        x(i + dummy*irep)=1.0d0
#else
        x(i)=1.0d0
#endif
      end do
    end do
    time2 = second()

    ns=1
!
! Testing 1 stream store with prefetch directive
!
#if MACHINE==SUNHPC || MACHINE==ITANIUM1 || MACHINE==MCKINLEY
  endif
#else
  elseif (nstreams .eq. 1)then
    do irep=1,nrep
#if MACHINE==T3E || MACHINE==TCS1 || MACHINE==TICK
     do i=1,n-7,8
        x(i) = 1.0
        x(i+1) = 1.0
        x(i+2) = 1.0
        x(i+3) = 1.0
        x(i+4) = 1.0
        x(i+5) = 1.0
        x(i+6) = 1.0
        x(i+7) = 1.0
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      do i=1,n
!IBM* PREFETCH_BY_LOAD(x(i+poff))
        x(i)=1.d0
#endif
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1=second()

    do irep=1,nrep
#if MACHINE==T3E || MACHINE==TCS1 || MACHINE==TICK
     do i=1,n-7,8
        x(i) = 1.0
        x(i+1) = 1.0
        x(i+2) = 1.0
        x(i+3) = 1.0
        x(i+4) = 1.0
        x(i+5) = 1.0
        x(i+6) = 1.0
        x(i+7) = 1.0
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      do i=1,n
!IBM* PREFETCH_BY_LOAD(x(i+poff))
        x(i)=1.d0
#endif
      end do
    end do
    time2=second()
!
! Testing 2 stream store with prefetch directive
!
  elseif(nstreams .eq. 2)then
    n = n/2
    do irep=1,nrep
#if MACHINE==T3E || MACHINE==TCS1 || MACHINE==TICK
      do i=1,n-3,4
        x(i)   = 1.0
        x(i+1) = 1.0
        x(i+2) = 1.0
	x(i+3) = 1.0
        x(i+noff+l2off)   = 1.d0
        x(i+1+noff+l2off) = 1.d0
        x(i+2+noff+l2off) = 1.d0
        x(i+3+noff+l2off) = 1.d0
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      do i=1,n
!IBM* PREFETCH_BY_LOAD(x(i+poff))
!IBM* PREFETCH_BY_LOAD(x(i+noff+l2off+l1off+poff))
        x(i) = 1.0
        x(i+noff+l2off+l1off)=1.d0
#endif
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1=second()

    do irep=1,nrep
#if MACHINE==T3E || MACHINE==TCS1 || MACHINE==TICK
      do i=1,n-3,4
        x(i)   = 1.0
        x(i+1) = 1.0
        x(i+2) = 1.0
        x(i+3) = 1.0
        x(i+noff+l2off)   = 1.d0
        x(i+1+noff+l2off) = 1.d0
        x(i+2+noff+l2off) = 1.d0
        x(i+3+noff+l2off) = 1.d0
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      do i=1,n
!IBM* PREFETCH_BY_LOAD(x(i+poff))
!IBM* PREFETCH_BY_LOAD(x(i+noff+l2off+l1off+poff))
        x(i) = 1.0
        x(i+noff+l2off+l1off)=1.d0
#endif
      end do
    end do

    time2=second()

!
! Testing 3 stream store with prefetch directive
!
  elseif(nstreams == 3)then
    ntemp = n/3/8   !align with cache line
    n = ntemp*8
    do irep=1,nrep
#if MACHINE==T3E || MACHINE==TCS1 || MACHINE==TICK
      do i=1,n
        x(i)   = 1.0
        x(i+noff+l2off)    = 1.d0
        x(i+2*(noff+l2off)) = 1.0d0
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      do i=1,n
!IBM* PREFETCH_BY_LOAD(x(i+poff))
!IBM* PREFETCH_BY_LOAD(x(i+noff+l2off+l1off+poff))
!IBM* PREFETCH_BY_LOAD(x(i+2*(noff+l2off+l1off)+poff))
        x(i)=1.d0
        x(i+noff+l2off+l1off)=1.d0
        x(i+2*(noff+l2off+l1off))=1.d0
#endif
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1=second()

    do irep=1,nrep
#if MACHINE==T3E || MACHINE==TCS1 || MACHINE==TICK
      do i=1,n
        x(i)   = 1.0
        x(i+noff+l2off)    = 1.d0
        x(i+2*(noff+l2off)) = 1.0d0
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      do i=1,n
!IBM* PREFETCH_BY_LOAD(x(i+poff))
!IBM* PREFETCH_BY_LOAD(x(i+noff+l2off+l1off+poff))
!IBM* PREFETCH_BY_LOAD(x(i+2*(noff+l2off+l1off)+poff))
        x(i)=1.d0
        x(i+noff+l2off+l1off)=1.d0
        x(i+2*(noff+l2off+l1off))=1.d0
#endif
      end do
    end do
    time2=second()

!
! Testing 4 stream store with prefetch directive
!
  elseif(nstreams == 4)then
    ntemp = n/4/8
    n = ntemp*8
    do irep=1,nrep
#if MACHINE==T3E || MACHINE==TCS1 || MACHINE==TICK
      do i=1,n
        x(i)   = 1.0
        x(i+noff+l2off)    = 1.d0
        x(i+2*(noff+l2off)) = 1.0d0
        x(i+3*(noff+l2off)) = 1.0d0
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      do i=1,n
!IBM* PREFETCH_BY_LOAD(x(i+poff))
!IBM* PREFETCH_BY_LOAD(x(i+noff+l2off+l1off+poff))
!IBM* PREFETCH_BY_LOAD(x(i+2*(noff+l2off+l1off)+poff))
!IBM* PREFETCH_BY_LOAD(x(i+3*(noff+l2off+l1off)+poff))
        x(i)=1.d0
        x(i+noff+l2off+l1off)=1.d0
        x(i+2*(noff+l2off+l1off))=1.d0
        x(i+3*(noff+l2off+l1off))=1.d0
#endif
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1=second()

    do irep=1,nrep
#if MACHINE==T3E || MACHINE==TCS1 || MACHINE==TICK
      do i=1,n
        x(i)   = 1.0
        x(i+noff+l2off)    = 1.d0
        x(i+2*(noff+l2off)) = 1.0d0
        x(i+3*(noff+l2off)) = 1.0d0
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      do i=1,n
!IBM* PREFETCH_BY_LOAD(x(i+poff))
!IBM* PREFETCH_BY_LOAD(x(i+noff+l2off+l1off+poff))
!IBM* PREFETCH_BY_LOAD(x(i+2*(noff+l2off+l1off)+poff))
!IBM* PREFETCH_BY_LOAD(x(i+3*(noff+l2off+l1off)+poff))
        x(i)=1.d0
        x(i+noff+l2off+l1off)=1.d0
        x(i+2*(noff+l2off+l1off))=1.d0
        x(i+3*(noff+l2off+l1off))=1.d0
#endif
      end do
    end do
    time2=second()

  elseif(nstreams == 5)then
    ntemp = n/5/8
    n = n*8
    do irep=1,nrep
      do i=1,n
        x(i)   = 1.0
        x(i+noff+l2off)    = 1.d0
        x(i+2*(noff+l2off)) = 1.0d0
        x(i+3*(noff+l2off)) = 1.0d0
        x(i+4*(noff+l2off)) = 1.0d0
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif
    time1 = second()
    do irep=1,nrep
      do i=1,n
        x(i)   = 1.0
        x(i+noff+l2off)    = 1.d0
        x(i+2*(noff+l2off)) = 1.0d0
        x(i+3*(noff+l2off)) = 1.0d0
        x(i+4*(noff+l2off)) = 1.0d0
      end do
    end do
    time2 = second()
  elseif(nstreams == 6)then
    ntemp = n/6/8
    n = ntemp*8
    do irep=1,nrep
      do i=1,n
        x(i)   = 1.0
        x(i+noff+l2off)    = 1.d0
        x(i+2*(noff+l2off)) = 1.0d0
        x(i+3*(noff+l2off)) = 1.0d0
        x(i+4*(noff+l2off)) = 1.0d0
        x(i+5*(noff+l2off)) = 1.0d0
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif
    time1=second()
    do irep=1,nrep
      do i=1,n
        x(i)   = 1.0
        x(i+noff+l2off)    = 1.d0
        x(i+2*(noff+l2off)) = 1.0d0
        x(i+3*(noff+l2off)) = 1.0d0
        x(i+4*(noff+l2off)) = 1.0d0
        x(i+5*(noff+l2off)) = 1.0d0
      end do
    end do
    time2=second()
  end if
#endif
  t=time2-time1

#if MODE==PARALLEL
!
! Do a reduce to see what was the max time by any one PE
!
  CALL MPI_ALLREDUCE(t,totalt,1,MPI_DOUBLE_PRECISION,&
              &   MPI_MAX,MPI_COMM_WORLD,ierr)
  t = totalt
#elif MODE==SERIAL
  totalt=t
#endif
!
! print out results
!
  if (iam .eq. 0) then
    t = totalt
    cyc = t*cpu_MHz/n/nrep/ns
    if(t .eq. 0 ) then
	mb = 0.0
    else
	mb = (float(nrep)*n*8*ns/t)/1E6
    endif
!   call display_screen(n,t,cyc,mb,'store')
  end if
  if (nstreams == 2) n=n*2
  if (nstreams == 3) n=n*3
  if (nstreams == 4) n=n*4
  if (nstreams == 5) n=n*5
  if (nstreams == 6) n=n*6

  return
  end subroutine store


!************************************************************
!***************    SUBROUTINE LOAD   ***********************
!************************************************************
  subroutine load(nrep,n,y,x,index,nstreams,noff,l2off,l1off,&
                &	step,iam,cyc,mb,t,cpu_MHz)
!
! This routine measures load bandwidth of the different levels of memory.
! Which level is tested depends on the value of n.  If n<size of L1 cache
! then mainly the bandwidth of L1 cache is tested. If L1 size<n<L2 size then
! mainly the bandwidth of L2 cache is tested. If L2 size<n<memory then mainly
! main memory is tested. Note that I say mainly because bandwidth test is sensitive
! to the size of n and its size relative to the different levels of memory.
!
  implicit none
#if MODE==PARALLEL
  include 'mpif.h'
  integer ierr
#endif
  integer it,irep,i,nrep,n,noff,l2off,l1off 
  integer nstreams,iam
  real*8 s1,s2,s3,s4,s5,s6,s7,s8
#if MACHINE==T3E
  real*8 y(16),x((4+8)*3*4*1024),cyc,mb,cpu_MHz
#else
  real*8 y(16),x((4+8)*noff),cyc,mb,cpu_MHz
#endif
  real*8 second,time1,time2,t,totalt
  integer off1, off2, off3,ntemp
  logical step
  integer*4 index(12*noff)
#if HPM==COUNT
  integer counter, hpmflag
  common /cnt/counter,hpmflag
  print *,'In Load counter=',counter
#endif

!
! If the step size is true then call another routine with similar loops
! to those below only with variable indexing.  This variable indexing
! determines bandwidth for stride-random loads.
!
  if (nrep <= 0 ) nrep=2

  if (step) then
    call load_random(nrep,n,y,x,index,nstreams,noff,&
    &           l2off,l1off,iam,cyc,mb,t,cpu_MHz)
    return
  end if

   s1=0.0
   s2=0.0
   s3=0.0
   s4=0.0
   s5 = 0.0
   s6 = 0.0
   s7 = 0.0
   s8 =0.0

! Testing loads for 1 stream (no need for prefetch directives because
! the system will recognize the need for 1 stream.
!
  if (nstreams == 1 ) then   
#if MACHINE==ITANIUM1 || MACHINE==MCKINLEY
    nstreams = 4
    n=n/4
!!!!
!!!! ITANIUM
    off1 = noff + 8*2*1024 + 2*256
    off2 = 2*noff + 16*2*1024 + 2*512
    off3 = 3*noff + 24*2*1024 + 2*768
!!!
!!! MCK
!    off1 =   noff +   1024 + 8*1024
!    off2 = 2*noff + 2*1024 + 16*1024
!    off3 = 3*noff + 3*1024 + 24*1024
#endif
#if MACHINE==TICK
    nstreams = 2
    n = n/2
#endif
    do irep=1,nrep
#if MACHINE==T3E || MACHINE==TCS1 
      do i = 1,n-1,2
        s1 = s1 + x(i)
        s2 = s2 * x(i+1)
#elif MACHINE==TICK
      do i = 1,n
        s1 = s1 + x(i)
        s2 = s2 * x(i+l2off)
#elif MACHINE==IBM_BH || MACHINE==SUNHPC 
      do i=1,n
        y(1) = y(1) + x(i)
#elif MACHINE==IBM_LH
      do i=1,n
        s1 = s1 + x(i)
#elif MACHINE==ITANIUM1 || MACHINE==MCKINLEY
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
#endif
        end do
     end do

#if MODE==PARALLEL
     call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

     time1 = second()

#if HPM==COUNT
    if (hpmflag==1) counter=counter+1
    if (hpmflag==1) call f_hpmstart(counter,"L1 loads")
#endif

     do irep=1,nrep
#if MACHINE==T3E || MACHINE==TCS1 
      do i = 1,n-1,2 
        s1 = s1 + x(i)
        s2 = s2 * x(i+1)
#elif MACHINE==TICK
      do i = 1,n
        s1 = s1 + x(i)
        s2 = s2 + x(i + l2off)
#elif MACHINE==IBM_BH || MACHINE==SUNHPC 
      do i=1,n
        y(1) = y(1) + x(i)
#elif MACHINE==IBM_LH
      do i=1,n
        s1 = s1 + x(i)
#elif MACHINE==ITANIUM1 || MACHINE==MCKINLEY
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
#endif
       end do
     end do

#if HPM==COUNT
    if (hpmflag==1) call f_hpmstop(counter)
#endif

     time2 = second()

! ONLY T3E and IBM testing for streams
#if MACHINE==T3E
  elseif(nstreams == 2)then
#elif MACHINE==IBM_BH 
  elseif(nstreams == 2 .AND. n > 8192)then
     off1 = noff+l2off+l1off
#elif MACHINE==IBM_LH
  elseif(nstreams == 2 .AND. n > 4096)then
     off1 = l2off
#endif

#if MACHINE==SUNHPC || MACHINE==ITANIUM1 || MACHINE==TICK || MACHINE==MCKINLEY || MACHINE==TCS1
  endif
#else
    n = n/2
    do irep=1,nrep
#if MACHINE==T3E
      do i=1,n-1,2
        s1 = s1 + x(i) +x(i+noff+l1off)
        s2 = s2 + x(i+1)*x(i+1+noff+l1off)
#elif MACHINE==IBM_BH 
      do i=1,n
        y(1)=y(1)+x(i)*x(i+off1)
#elif MACHINE==IBM_LH
      do i=1,n
        s1 = s1 + x(i)*x(i+off1)
#endif
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1=second()
#if HPM==COUNT
    if (hpmflag==1) counter=counter+1
    if (hpmflag==1) call f_hpmstart(counter,"L1 loads")
#endif
    do irep=1,nrep
#if MACHINE==T3E
      do i=1,n-1,2
        s1 = s1 + x(i) +x(i+noff+l1off)
        s2 = s2 + x(i+1)*x(i+1+noff+l1off)
#elif MACHINE==IBM_BH || MACHINE==IBM_LH
      do i=1,n
        y(1)=y(1)+x(i)*x(i+off1)
#endif
      end do
    end do
#if HPM==COUNT
    if (hpmflag==1) call f_hpmstop(counter)
#endif

    time2=second()

!
! Testing loads on L1 cache.  This is because you need to make the code
! memory bound not compute bound so inner core of the loop needs to be
! more than just a single load.
!
#if MACHINE==IBM_BH 
  elseif(nstreams == 2 .AND. n <= 8192)then
    n = n/2
    do irep=1,nrep
      do i=1,n
        y(1)=y(1)+x(i)*x(i+noff+l2off+l1off)
      end do
    end do
#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1 = second()
#if HPM==COUNT
    if (hpmflag==1) counter=counter+1
    if (hpmflag==1) call f_hpmstart(counter,"L1 loads")
#endif
    do irep=1,nrep
      do i=1,n
        y(1)=y(1)+x(i)*x(i+noff+l2off+l1off)
      end do
    end do
#if HPM==COUNT
    if (hpmflag==1) call f_hpmstop(counter)
#endif

    time2 = second()
#endif

#if MACHINE==IBM_LH
  elseif(nstreams == 4 .AND. n <= 4096)then
    off1 =   noff + 22*5.625*1024+4.25*1024+1.375*1024-896
    off2 = 2*noff + 45*5.625*1024+2.875*1024+2.75*1024+256
    off3 = 3*noff + 68*5.625*1024+1.5*1024+4.125*1024+1.375*1024
    n = n/4
    do irep=1,nrep
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
      end do
    end do
#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif
    time1 = second()
    do irep=1,nrep
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
      end do
    end do
    time2 = second()
#endif
!
! Testing loads for 3 streams.
!
  elseif(nstreams == 3)then
    ntemp = n/3/8
    n = ntemp*8
#if MACHINE==IBM_LH
    off1 =   noff +22*7.5*1024+5*1024+2.5*1024-384
    off2 = 2*noff + 22*15*1024+10*1024+5*1024+0.25*1024
#endif
    do irep=1,nrep
#if MACHINE==T3E
      do i=1,n
       s1=s1+x(i) 
       s2=s2+x(i+noff+l1off)*x(i+2*(noff+l1off))
#elif MACHINE==IBM_BH
      do i=1,n
        y(1)=y(1)+x(i)+x(i+noff+l2off+l1off)&
     &        +x(i+2*(noff+l2off+l1off))
#elif MACHINE==IBM_LH
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)
        s6 = s6 + x(i+1+off2)
        s7 = s7 + x(i+2+off2)
        s8 = s8 + x(i+3+off2)
#endif
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1=second()
#if HPM==COUNT
    if (hpmflag==1) counter=counter+1
    if (hpmflag==1) call f_hpmstart(counter,"L1 loads")
#endif

    do irep=1,nrep
#if MACHINE==T3E
      do i=1,n
       s1=s1+x(i)
       s2=s2+x(i+noff+l1off)*x(i+2*(noff+l1off))
#elif MACHINE==IBM_BH
      do i=1,n
        y(1)=y(1)+x(i)+x(i+noff+l2off+l1off)&
     &        +x(i+2*(noff+l2off+l1off))
#elif MACHINE==IBM_LH
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)
        s6 = s6 + x(i+1+off2)
        s7 = s7 + x(i+2+off2)
        s8 = s8 + x(i+3+off2)
#endif
      end do
    end do
#if HPM==COUNT
    if (hpmflag==1) call f_hpmstop(counter)
#endif

    time2=second()

!
! Testing loads for 4 streams.
!
  elseif(nstreams == 4)then
    ntemp = n/4/8
    n = ntemp*8
#if MACHINE==T3E  
    off1 = noff+l1off
    off2 = 2*(noff+l1off)
    off3 = 3*(noff+l1off)
#elif MACHINE==IBM_BH
    off1 = noff+l2off+l1off
    off2 = 2*(noff+l2off+l1off)
    off3 = 3*(noff+l2off+l1off)
#elif MACHINE==IBM_LH
    off1 =   noff + 22*5.625*1024+4.25*1024+1.375*1024-896
    off2 = 2*noff + 45*5.625*1024+2.875*1024+2.75*1024+256
    off3 = 3*noff + 68*5.625*1024+1.5*1024+4.125*1024+1.375*1024
#endif
    do irep=1,nrep
#if MACHINE==T3E
      do i=1,n
        s1 = s1 +x(i)+x(i+off1)
        s2 = s2 +x(i+off2)*x(i+off3)
#elif MACHINE==IBM_BH 
      do i=1,n
        y(1)=y(1)+x(i)*x(i+off1)
        y(2)=y(2)+x(i+off2)*x(i+off3)
#elif MACHINE==IBM_LH
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
#endif
      end do
    end do  

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1=second()
#if HPM==COUNT
    if (hpmflag==1) counter=counter+1
    if (hpmflag==1) call f_hpmstart(counter,"L1 loads")
#endif
    do irep=1,nrep
#if MACHINE==T3E
      do i=1,n
        s1 = s1 +x(i)+x(i+off1)
        s2 = s2 +x(i+off2)*x(i+off3)
#elif MACHINE==IBM_BH 
      do i=1,n
        y(1)=y(1)+x(i)*x(i+off1)
        y(2)=y(2)+x(i+off2)*x(i+off3)
#elif MACHINE==IBM_LH
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
#endif
      end do
    end do  
#if HPM==COUNT
    if (hpmflag==1) call f_hpmstop(counter)
#endif

    time2=second()

#if MACHINE==T3E
  elseif(nstreams == 5)then
    ntemp = n/5/8
    n = ntemp*8
    do irep=1,nrep
      do i=1,n
        s1 = s1 +x(i)+x(i+noff+l1off)+x(i+4*(noff+l1off))&
        &+x(i+2*(noff+l1off))+x(i+3*(noff+l1off))
      end do
    end do 
#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif
    time1 = second()
    do irep=1,nrep
      do i=1,n
        s1 = s1 +x(i)+x(i+noff+l1off)+x(i+4*(noff+l1off))&
        &+x(i+2*(noff+l1off))+x(i+3*(noff+l1off))
      end do
    end do 
    time2 = second()
  elseif(nstreams == 6)then
    ntemp = n/6/8
    n = ntemp*8
    do irep=1,nrep
      do i=1,n
        s1 = s1 +x(i)+x(i+noff+l1off)+x(i+4*(noff+l1off))+x(i+2*(noff+l1off))
        s2 = s2+x(i+3*(noff+l1off))*x(i+5*(noff+l1off))
      end do
    end do
#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif
    time1 = second()
    do irep=1,nrep
      do i=1,n
        s1 = s1 +x(i)+x(i+noff+l1off)+x(i+4*(noff+l1off))+x(i+2*(noff+l1off))
        s2 = s2+x(i+3*(noff+l1off))*x(i+5*(noff+l1off))
      end do
    end do
    time2 = second()
!end if for T3E for 5-6 streams
#endif
  endif

!endif for all not testing streams
#endif
	
  t=time2-time1
  if (t == 0.0) then
        print *,'stopping due to time=0'
	stop
  endif
#if MODE==PARALLEL
!
! Do a reduce to see what was the max time by any one PE
!
  CALL MPI_ALLREDUCE(t,totalt,1,MPI_DOUBLE_PRECISION,&
        &         MPI_MAX,MPI_COMM_WORLD,ierr)
  t = totalt
#elif MODE==SERIAL
  totalt=t
#endif
!
! print out results
!
  y(1)=y(1)+ s1 +s2 +s3 +s4+s5+s6+s7+s8
 
  if (iam .eq. 0) then
    t = totalt
    cyc = t*cpu_MHz/n/nrep/nstreams
    mb = (float(nrep)*n*8*nstreams/t)/1E6
   !call display_screen(n,t,cyc,mb,'load ')
  end if
  if (nstreams == 2) n=n*2
  if (nstreams == 3) n=n*3
  if (nstreams == 4) n=n*4
  if (nstreams == 5) n=n*5
  if (nstreams == 6) n=n*6
#if MACHINE==ITANIUM1 || MACHINE==TICK || MACHINE==MCKINLEY
  nstreams=1
#endif
  return
  end subroutine load

 
!************************************************************
!***************    SUBROUTINE STORE_S   ********************
!************************************************************
  subroutine store_random(nrep,n,y,x,index,nstreams,noff,l2off,l1off,&
              &        poff,iam,cyc,mb,t,cpu_MHz)
  implicit none
#if MODE==PARALLEL
  include 'mpif.h'
  integer ierr
#endif
  integer it,irep,i,nrep,n,noff,l2off,l1off,poff
  integer nstreams,iam,off
  real*8 y(16),x((4+8)*noff),cyc,mb,cpu_MHz,s,z
  real*8 time1,time2,second,t,totalt
  integer*4 index(12*noff)
  integer ran_num,j,flag
  logical not_picked
!
! value of off is set so that the index array does not conflict
! in any level of cache with the testing array.
!
#if MACHINE==T3E
  off = noff+l2off*2
#elif MACHINE==IBM_BH
  off = noff+l2off+8192*2
#elif MACHINE==IBM_LH
  off = noff + l2off
#elif MACHINE==TCS1 || MACHINE==ITANIUM1 || MACHINE==TICK || MACHINE==MCKINLEY || MACHINE==SUNHPC
  off = noff+l2off
#endif
!
! generate the index array so that you step through
! memory randomly
!
  index =0
  call random_number(z)
  index(off+1) = int(real(n)*z)+1
  do i = off+2,n+off
    call random_number(z)
    ran_num = int(real(n)*z)+1
    index(i) = ran_num
  end do

  if (nstreams .eq. 0) then
    do irep=1,nrep
      do i=1,n
        x(index(i))=1.d0
      end do
    end do
#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif
    time1 = second()
    do irep=1,nrep
      do i=1,n-3,4
        x(index(i))=1.d0
	x(index(i+1))=1.d0
	x(index(i+2))=1.d0
	x(index(i+3))=1.d0
      end do
    end do
    time2 = second()
    nstreams=1
  elseif (nstreams .eq. 1)then
    do irep=1,nrep
      do i=off+1,off+n-3,4
        x(index(i))=1.d0
        x(index(i+1))=1.d0
        x(index(i+2))=1.d0
        x(index(i+3))=1.d0
      end do
    end do
#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1 = second()
    do irep=1,nrep
      do i=off+1,off+n-3,4
        x(index(i))=1.d0
        x(index(i+1))=1.d0
        x(index(i+2))=1.d0
        x(index(i+3))=1.d0
      end do
    end do
    time2 = second()
  end if
  t=time2-time1

#if MODE==PARALLEL
  CALL MPI_ALLREDUCE(t,totalt,1,MPI_DOUBLE_PRECISION,&
          &  MPI_MAX,MPI_COMM_WORLD,ierr)
  t = totalt
#elif MODE==SERIAL
  totalt=t
#endif
!
! print out results
!
  if (iam .eq. 0) then
    t = totalt
    cyc = t*cpu_MHz/n/nrep/nstreams
    mb = (float(nrep)*n*8*nstreams/t)/1E6
   !call display_screen(n,t,cyc,mb,'store')
  end if
  return
  end subroutine store_random


!************************************************************
!************    SUBROUTINE LOAD_RANDOM   *******************
!************************************************************
  subroutine load_random(nrep,n,y,x,index,nstreams,noff,l2off,l1off,&
                    &	iam,cyc,mb,t,cpu_MHz)
  implicit none
#if MODE==PARALLEL
  include 'mpif.h'
  integer ierr
#endif
  integer it,irep,i,nrep,n,noff,l2off,l1off
  integer nstreams,iam
  real*8 y(16),x((4+8)*noff),cyc,mb,cpu_MHz,z
  real*8 t,totalt,second,time1,time2
  real*8 s1,s2,s3,s4,s5,s6,s7,s8
#if MACHINE==ITANIUM1 || MACHINE==MCKINLEY || MACHINE==IBM_LH
  integer off1,off2,off3
#endif
  integer*4 index(12*noff)
  integer ran_num,j,flag,off,ntemp
  logical not_picked
#if HPM==COUNT
  integer counter, hpmflag
  common /cnt/counter,hpmflag
  print *,'In ran Load counter=',counter
#endif
  s1=0.0
  s2=0.0
  s3=0.0
  s4=0.0
  s5=0.0
  s6=0.0
  s7=0.0
  s8=0.0
!
! value of off is set so that the index array does not conflict
! in any level of cache with the testing array.
!
#if MACHINE==T3E
  off = noff+l2off*2
#elif MACHINE==IBM_BH
  off = noff+l2off+8192*2
#elif MACHINE==IBM_LH
  off = noff+l2off
#elif MACHINE==TCS1 || MACHINE==SUNHPC || MACHINE==TICK
  off = noff+l2off
#elif MACHINE==ITANIUM1 
  off1 = noff   + 7*2*1024 + 448
  off2 = 2*noff + 14*2*1024+ 896
  off3 = 3*noff + 21*2*1024+ 1344
  off =  4*noff + 28*2*1024+ 1792
  n = n/2
#elif MACHINE==MCKINLEY
  n = n/2
  off1 =   noff + 13104 + 1632 !2streams
  off  = 2*noff + 26208 + 3264 !2streams
!  off1 =   noff +  7280 +  912 !4streams
!  off2 = 2*noff + 14560 + 1824 !4streams
!  off3 = 3*noff + 21840 + 2736 !4streams
!  off  = 4*noff + 29120 + 3648 !4streams
#endif
!
! generate the index array so that you step through
! memory randomly
!
  index =0
  call random_number(z)
  index(off+1) = int(real(n)*z)+1
  do i = off+2,n+off
     call random_number(z)
     ran_num = int(real(n)*z)+1
     index(i) = ran_num
  end do
  it=1   
  if (nstreams .eq. 1) then
     do irep=1,nrep
#if MACHINE==ITANIUM1 || MACHINE==MCKINLEY
       do i = 1+off,n+off-3,4
         s1 = s1 +x(index(i))*x(index(i)+off1)
         s2 = s2 +x(index(i+1))*x(index(i+1)+off1)
         s3 = s3 +x(index(i+2))*x(index(i+2)+off1)
         s4 = s4 +x(index(i+3))*x(index(i+3)+off1)
!         s5 = s5 +x(index(i)+off2)*x(index(i)+off3)
!         s6 = s6 +x(index(i+1)+off2)*x(index(i+1)+off3)
!         s7 = s7 +x(index(i+2)+off2)*x(index(i+2)+off3)
!         s8 = s8 +x(index(i+3)+off2)*x(index(i+3)+off3)
#else
       do i = 1+off,n+off-7,8
         y(1) = y(1)+x(index(i))
         y(2) = y(2) *x(index(i+1))
         y(3) = y(3) +x(index(i+2))
         y(4) = y(4) *x(index(i+3))
         y(5) = y(5) +x(index(i+4))
         y(6) = y(6) *x(index(i+5))
         y(7) = y(7) +x(index(i+6))
         y(8) = y(8) *x(index(i+7))
#endif
       end do
     end do
#if MODE==PARALLEL
     call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif
     time1  = second()

#if HPM==COUNT
    if (hpmflag==1) counter=counter+1
    if (hpmflag==1) call f_hpmstart(counter,"L1 loads")
#endif
     do irep=1,nrep
#if MACHINE==ITANIUM1 || MACHINE==MCKINLEY
       do i=1+off,n+off-3,4
         s1 = s1 +x(index(i))*x(index(i)+off1)
         s2 = s2 +x(index(i+1))*x(index(i+1)+off1)
         s3 = s3 +x(index(i+2))*x(index(i+2)+off1)
         s4 = s4 +x(index(i+3))*x(index(i+3)+off1)
!         s5 = s5 +x(index(i)+off2)*x(index(i)+off3)
!         s6 = s6 +x(index(i+1)+off2)*x(index(i+1)+off3)
!         s7 = s7 +x(index(i+2)+off2)*x(index(i+2)+off3)
!         s8 = s8 +x(index(i+3)+off2)*x(index(i+3)+off3)
#else
       do i=1+off,n+off-7,8
         y(1) = y(1)+x(index(i))
         y(2) = y(2) *x(index(i+1))
         y(3) = y(3) +x(index(i+2))
         y(4) = y(4) *x(index(i+3))
         y(5) = y(5) +x(index(i+4))
         y(6) = y(6) *x(index(i+5))
         y(7) = y(7) +x(index(i+6))
         y(8) = y(8) *x(index(i+7))
#endif
       end do
     end do
#if HPM==COUNT
    if (hpmflag==1) call f_hpmstop(counter)
#endif
     time2 = second()
  endif

#if MACHINE==ITANIUM1 || MACHINE==MCKINLEY
  y(1) = s1+s2+s3+s4+s5+s6+s7+s8
  nstreams=2
#endif
  t = time2-time1
#if MODE==PARALLEL
  CALL MPI_ALLREDUCE(t,totalt,1,MPI_DOUBLE_PRECISION,&
           &  MPI_MAX,MPI_COMM_WORLD,ierr)
  t = totalt
#elif MODE==SERIAL
  totalt=t
#endif
!
! print out results
!
  if (iam .eq. 0) then
    t = totalt
    cyc = t*cpu_MHz/n/nrep/nstreams
    mb = (float(nrep)*n*8*nstreams/t)/1E6
   !call display_screen(n,t,cyc,mb,'load ')
  end if
#if MACHINE==ITANIUM1 || MACHINE==MCKINLEY
  n=n*2
  nstreams=1
#endif

  return
  end subroutine load_random


!************************************************************
!***************    SUBROUTINE DISPLAY   ********************
!************************************************************
  subroutine display(name,type,step,tests,ns,n)
  implicit none
  integer tests,ns,n
  logical step
  character(LEN=12) name
  character(LEN=5) type
!
! display to screan what is being tested
!
  if (step)then
    write(6,*)'Testing n=',n,' for ',name,' ',type,&
	    &' stepping randomly for',&
            &	tests,' tests and for ',ns,' streams.'
  else
    write(6,*)'Testing n=',n,' for ',name,' ',type,&
	    &  ' stepping by one for',&
            &      tests,' tests and for ',ns,' streams.'
  end if
  return
  end subroutine display


!************************************************************
!***************    SUBROUTINE DISPLAY_SMALL   **************
!************************************************************
  subroutine display_small(name,type,n,ns,step,cyc,mb,tests,unit)
  implicit none
  integer ns,n,tests,i,unit
  real*8 cyc(tests),mb(tests),small,megab
  character(LEN=12) name
  character(LEN=5) type
  character*4 ran
  logical step
#if HPM==COUNT
  integer counter, hpmflag
  common /cnt/counter,hpmflag
  print *,'In DISPLAY counter=',counter
#endif

  megab=0.0
!.
!. find the smallest cyc  and largest mb of all tests
!.
  do i=1,tests
   if (mb(i) .gt. megab) then
     small=cyc(i)
     megab = mb(i)
   end if
  end do
  if (step)then
    ran="R"
    n=(n/2)*3
  else
    ran="NR"
  endif

  write(unit,100)name,type,n,ns,ran,small,megab
100     format('**',A12,',',A5,',',I10,',',I11,',',A4,',',E12.4,',',F10.4)
  return
  end subroutine display_small


!************************************************************
!************    SUBROUTINE DISPLAY_SCREEN   ****************
!************************************************************
  subroutine display_screen(n,t,cyc,mb,type)
  implicit none
  character(LEN=5) type
  real*8 cyc,mb
  real*8 t
  integer n

!
! display to screen the results of each test
!
  write(6,FMT='(A6,F10.4)')'time=',t
  write(6,FMT='(A12,A5,A1,F12.8)')'cycles per ',type,':',cyc
  write(6,FMT='(A17,F10.4)')'Mbytes per sec :',mb
  write(6,*)
  return
  end subroutine display_screen



!************************************************************
!**************    SUBROUTINE SETUP_TEST   ******************
!************************************************************
  subroutine setup_test(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
          &  name,L1s,L2s,L3s,type,tests,subnam,recommended_time,cpu_MHz)
  implicit none
  character(LEN=12) name
  character(LEN=5) type

  integer nrep,n,noff,l2off,l1off
  integer ns,L1s,L2s,L3s,iam,tests
  real*8 y(16),x((4+8)*noff),cyc,mb
  logical step
  logical value_set
  real*8 recommended_time,cpu_MHz,time
  integer*4 index(12*noff)
  external subnam
#if HPM==COUNT
  integer counter, hpmflag
  common /cnt/counter,hpmflag
#endif
! 
! The purpose of this test is to determine the right value 
! of nrep so the test time is in specified range.
!

#if MACHINE==T3E || MACHINE==TCS1 
  if ((n .le. L2s)  ) then
     if (n .le. L1s) name='L1 cache'
     if ((n .le. L2s) .and. (n .gt. L1s) ) name="L2 cache"
     if ((type .ne. "Store") .and. (.not. step) ) then
	ns=1
	return
     else
	ns=1
     end if
#else
  if (n .le. L1s) then
#endif
    name='L1 cache'
#if MACHINE==IBM_BH 
    ns = 2
#elif MACHINE==IBM_LH
    ns=4
#elif MACHINE==SUNHPC || MACHINE==ITANIUM1 || MACHINE==TICK || MACHINE==MCKINLEY
    ns=1
#endif
    if (type .eq. "Store") ns=0
    if (step) ns=1
  elseif ((n .le. L2s) .and. (n .ge. L1s) )then
    name="L2 cache"
#if MACHINE==SUNHPC || MACHINE==ITANIUM1 || MACHINE==TICK || MACHINE==MCKINLEY
    ns=1 !NO streams
    if (type .eq. "Store") ns=0
#endif

#if MACHINE==IBM_LH
  elseif ((n .le. L3s) .and. (n .ge. L2s) )then
    name="L3 cache"
#endif
  else
    name="Main Memory"
#if MACHINE==TCS1  || MACHINE==TICK
    ns = 1  !assuming one stream for now
#elif MACHINE==SUNHPC 
    ns=1
    if (type .eq. "Store") ns=0
#elif MACHINE==ITANIUM1 || MACHINE==MCKINLEY
    ns=6
    if (type .eq. "Store") ns=0
#endif
  end if
  value_set=.true.
  nrep = nrep/2
#if MACHINE==MCKINLEY
  if (.not. step) return
#endif
  do while(value_set)
#if HPM==COUNT
    hpmflag=0
#endif
#if MACHINE==IBM_LH
    call load_lh(n,mb,cyc,time,nrep,ns,step,cpu_MHz)
#else
    call subnam(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
	&	cyc,mb,time,cpu_MHz)
#endif
    if (n == 0 ) stop
#if HPM==COUNT
    hpmflag=1
#endif
    if (time .gt. recommended_time) then
      if (nrep .le. 1)then
	value_set=.false.
      else
        nrep=nrep/2
      endif
    elseif( time .lt. 2.5 )then
      nrep=nrep*2
    else
      value_set=.false.
    endif
  end do
  return
  end subroutine setup_test



!************************************************************
!***************    SUBROUTINE TESTING   ********************
!************************************************************
  subroutine testing(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
     & name,L1s,L2s,L3s,type,tests,subnam,recommended_time,cpu_MHz,unit)
  implicit none
  character(LEN=12) name
  character(LEN=5) type
  integer nrep,n,noff,l2off,l1off,i,unit
  integer ns,L1s,L2s,L3s,iam,tests
  real*8 y(16),x((4+8)*noff),cyc,mb,cycles(tests),megab(tests)
  real*8 recommended_time,cpu_MHz,t
  logical step
  integer*4 index(12*noff)
  external subnam
!
! The purpose of this test is first to determine the right value
! of nrep so the test time is in specified range.  Once this is
! determined the test is run a number of times, the number specified
! by user in RunConditions file.
!
  call setup_test(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
     &   name,L1s,L2s,L3s,type,tests,subnam,recommended_time,cpu_MHz)
  if (iam .eq. 0) call display(name,type,step,tests,ns,n)

!
!   Loop through "tests" number of tests calling either load or store 
!   subroutine depending on value of subnam.  After all the tests call
!   display_small which will display the best result from all tests.
!
  do i=1,tests
#if MACHINE==T3E || MACHINE==TCS1
     if (( n .le. L2s) .and. (type .eq. 'Load ') .and. (.not. step) ) then
       call load_alpha(n,unit,mb,cyc,t)
     else
       call subnam(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
        &       cyc,mb,t,cpu_MHz)
     endif
#elif MACHINE==MCKINLEY
    if  (.not. step) then
       call load_mck(n,unit,mb,cyc,t)
    else
       call subnam(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
        &       cyc,mb,t,cpu_MHz)
    endif
#elif MACHINE==IBM_LH
    if  (.not. step) then
       call load_lh(n,mb,cyc,t,nrep,ns,step,cpu_MHz)
    else
       call subnam(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
        &       cyc,mb,t,cpu_MHz)
    endif
#else
    call subnam(nrep,n,y,x,index,ns,noff,l2off,l1off,step,iam,&
        &       cyc,mb,t,cpu_MHz)
#endif
    if (iam .eq. 0) call display_screen(n,t,cyc,mb,type)
    cycles(i)=cyc
    megab(i)=mb
  end do
  if(iam .eq. 0) then
    call display_small(name,type,n,ns,step,cycles,megab,tests,unit)
  end if

  end subroutine testing


!************************************************************
!**************    SUBROUTINE LOAD_ALPHA   ******************
!************************************************************
#if MACHINE==T3E || MACHINE==TCS1
  subroutine load_alpha(n,unit,mb,cyc,t)
  implicit none
#if MODE==PARALLEL
  include 'mpif.h'
  integer ierr
#endif
  character(LEN=12) name
  character(LEN=5) type
  integer irep,i,nrep,n,noff,l2off,l1off,unit
  integer poff,ns

#if MACHINE==T3E
  parameter(noff=3*4*1024, l2off=4*1024+512, l1off=8, poff=8)
#elif MACHINE==TCS1
  parameter(noff=1024*1024, l2off=128*4*1024+2*1024+8, l1off=8, poff=8)
#endif
  real*8 y(16),x(12*noff),sum,cyc,mb,pad(16),pad2(16)
  integer*4 index(12*noff)
  integer ntest,iam
  real*8 t,totalt,cpu_MHz,time1,time2,second

  common /arry/x,pad,y,pad2,index
!DIR$ CACHE_ALIGN /arry/

  call random_number(x)
  call random_number(y)
  ns = 2
!  n=256
!  nrep = 2000000
   nrep = 10000
#if MACHINE==T3E
  cpu_MHz = 300000000.0
#elif MACHINE==TCS1
  cpu_MHz = 667000000.0
#endif
  type='load'
  n = n/2
!
! Start memory bandwidth test
!
  do irep=1,nrep
    do i=1,n-3,4
      y(1) = y(1) + x(i)
      y(2) = y(2) * x(i+1)
      y(3) = y(3) + x(i+2)
      y(4) = y(4) * x(i+3)
      y(5) = y(5) + x(i+l2off)
      y(6) = y(6) * x(i+1+l2off)
      y(7) = y(7) + x(i+2+l2off)
      y(8) = y(8) * x(i+3+l2off)
    end do
  end do

  time1 = second()
  do irep=1,nrep
    do i=1,n-3,4
      y(1) = y(1) + x(i)
      y(2) = y(2) * x(i+1)
      y(3) = y(3) + x(i+2)
      y(4) = y(4) * x(i+3)
      y(5) = y(5) + x(i+l2off)
      y(6) = y(6) * x(i+1+l2off)
      y(7) = y(7) + x(i+2+l2off)
      y(8) = y(8) * x(i+3+l2off)
    end do
  end do
  time2 = second()
  
  t=time2 - time1

#if MODE==PARALLEL
!
! Do a reduce to see what was the max time by any one PE
!
  CALL MPI_ALLREDUCE(t,totalt,1,MPI_DOUBLE_PRECISION,&
        &         MPI_MAX,MPI_COMM_WORLD,ierr)
  t = totalt
#elif MODE==SERIAL
  totalt=t
  iam = 0
#endif

!
! print out results
!
  cyc = t*cpu_MHz/n/nrep/ns
  mb = (float(nrep)*n*8*ns/t)/1E6
!
  n = n*2
  return
  end subroutine load_alpha
#endif


!************************************************************
!***************    SUBROUTINE LOAD_LH   ********************
!************************************************************
#if MACHINE==IBM_LH
  subroutine load_lh(n,mb,cyc,t,nrep,nstreams,step,cpu_MHz)

!
! This routine measures load bandwidth of the different levels of memory.
! Which level is tested depends on the value of n.  If n<size of L1 cache
! then mainly the bandwidth of L1 cache is tested. If L1 size<n<L2 size then
! mainly the bandwidth of L2 cache is tested. If L2 size<n<memory then mainly
! main memory is tested. Note that I say mainly because bandwidth test is sensitive
! to the size of n and its size relative to the different levels of memory.
!
  implicit none
#if MODE==PARALLEL
  include 'mpif.h'
  integer ierr
#endif
  character(LEN=12) name
  character(LEN=5) type
  logical step
  real*8 s1,s2,s3,s4,s5,s6,s7,s8
  real*8 second, time1,time2
  integer off1, off2, off3, ntemp

  integer irep,i,nrep,n,noff,l2off,l1off
  integer nstreams,iam

  parameter(noff=2*512*1024, l2off=noff + 256*1024 + 11.25*1024 - 256)
  real*8 y(16),x(12*noff),sum,cyc,mb,pad(16),pad2(16)
  integer*4 index(12*noff)
  integer ntest
  real*8 t,totalt,cpu_MHz

  common /arry/x,pad,y,pad2,index
!DIR$ CACHE_ALIGN /arry/

#if HPM==COUNT
  integer counter, hpmflag
  common /cnt/counter,hpmflag
  print *,'In Load counter=',counter
#endif

!
! If the step size is true then call another routine with similar loops
! to those below only with variable indexing.  The separate routine is
! needed because if the same loops were used and index array was incremented
! by 1, it was seen that these loops did not perform as well as
! loops without index variable.  This is probably due to some compiler
! optimization.
!
!  if (step) then
!    call load_random(nrep,n,y,x,index,nstreams,hz,noff,&
!    &           l2off,l1off,iam,cyc,mb,t,cpu_MHz)
!    return
!  end if

   s1=0.0
   s2=0.0
   s3=0.0
   s4=0.0
   s5 = 0.0
   s6 = 0.0
   s7 = 0.0
   s8 =0.0

! Testing loads for 1 stream (no need for prefetch directives because
! the system will recognize the need for 1 stream.
!
  if (nstreams == 1 ) then
    do irep=1,nrep
      do i=1,n
        s1 = s1 + x(i)
        end do
     end do

#if MODE==PARALLEL
     call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

     time1 = second()

#if HPM==COUNT
    if (hpmflag==1) counter=counter+1
    if (hpmflag==1) call f_hpmstart(counter,"L1 loads")
#endif

     do irep=1,nrep
      do i=1,n
        s1 = s1 + x(i)
       end do
     end do

#if HPM==COUNT
    if (hpmflag==1) call f_hpmstop(counter)
#endif

     time2 = second()

  elseif(nstreams == 2 .AND. n > 4096)then
    off1 = l2off
    n = n/2
    do irep=1,nrep
      do i=1,n
        s1 = s1 + x(i)*x(i+off1)
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1=second()

    do irep=1,nrep
      do i=1,n
        s1=s1+x(i)*x(i+off1)
      end do
    end do

    time2=second()

!
! Testing loads on L1 cache.  This is because you need to make the code
! memory bound not compute bound so inner core of the loop needs to be
! more than just a single load.
!
  elseif(nstreams == 4 .AND. n <= 4096)then
    off1 =   noff + 22*5.625*1024+4.25*1024+1.375*1024-896
    off2 = 2*noff + 45*5.625*1024+2.875*1024+2.75*1024+256
    off3 = 3*noff + 68*5.625*1024+1.5*1024+4.125*1024+1.375*1024
    n = n/4
    do irep=1,nrep
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1 = second()
    do irep=1,nrep
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
      end do
    end do
    time2 = second()
!
! Testing loads for 3 streams.
!
  elseif(nstreams == 3)then
    ntemp = n/3/8
    n = ntemp*8
    off1 =   noff +22*7.5*1024+5*1024+2.5*1024-384
    off2 = 2*noff + 22*15*1024+10*1024+5*1024+0.25*1024
    do irep=1,nrep
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)
        s6 = s6 + x(i+1+off2)
        s7 = s7 + x(i+2+off2)
        s8 = s8 + x(i+3+off2)
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1=second()

    do irep=1,nrep
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)
        s6 = s6 + x(i+1+off2)
        s7 = s7 + x(i+2+off2)
        s8 = s8 + x(i+3+off2)
      end do
    end do
    time2=second()

!
! Testing loads for 4 streams.
!
  elseif(nstreams == 4)then
    ntemp = n/4/8
    n = ntemp*8
    off1 =   noff + 22*5.625*1024+4.25*1024+1.375*1024-896
    off2 = 2*noff + 45*5.625*1024+2.875*1024+2.75*1024+256
    off3 = 3*noff + 68*5.625*1024+1.5*1024+4.125*1024+1.375*1024
    do irep=1,nrep
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1=second()
    do irep=1,nrep
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
      end do
    end do
    time2=second()

  endif

  t=time2-time1
  if (t == 0.0) then
        print *,'stopping due to time=0'
        stop
  endif
#if MODE==PARALLEL
!
! Do a reduce to see what was the max time by any one PE
!
  CALL MPI_ALLREDUCE(t,totalt,1,MPI_DOUBLE_PRECISION,&
        &         MPI_MAX,MPI_COMM_WORLD,ierr)
  t = totalt
#elif MODE==SERIAL
  totalt=t
  iam = 0
#endif

! print out results
!
  y(1)=y(1)+ s1 +s2 +s3 +s4+s5+s6+s7+s8

  if (iam .eq. 0) then
    t = totalt
    cyc = t*cpu_MHz/n/nrep/nstreams
    mb = (float(nrep)*n*8*nstreams/t)/1E6
   !call display_screen(n,t,cyc,mb,'load ')
  end if
  if (nstreams == 2) n=n*2
  if (nstreams == 3) n=n*3
  if (nstreams == 4) n=n*4
  if (nstreams == 5) n=n*5
  if (nstreams == 6) n=n*6
  return
  end subroutine load_lh
#endif


!************************************************************
!***************    SUBROUTINE LOAD_MCK   *******************
!************************************************************
#if MACHINE==MCKINLEY
  subroutine load_mck(n,unit,mb,cyc,t)
  implicit none
#if MODE==PARALLEL
  include 'mpif.h'
  integer ierr
#endif
  character(LEN=12) name
  character(LEN=5) type
  real*8 s1,s2,s3,s4,s5,s6,s7,s8,time1,time2
  real*8 second,time12,time11
  integer off1, off2, off3

  integer irep,i,nrep,n,noff,l2off,l1off,unit
  integer poff,ns

  parameter(noff=2*3*128*1024, l2off=8*1024+1024, l1off=8,poff=8)
  real*8 y(16),x(12*noff),sum,cyc,mb,pad(16),pad2(16)
  integer*4 index(12*noff)
  integer hz,tstart,tstop
  integer ntest
  real*8 t,totalt,cpu_MHz

  common /arry/x,pad,y,pad2,index
!DIR$ CACHE_ALIGN /arry/

  call random_number(x)
  call random_number(y)
  ns = 4
  nrep = 1000000
  if (n > 21504) nrep = 100000
  if (n > 128800) nrep = 50000
  if (n > 297000) nrep = 5000
  if (n > 500000) nrep = 1000
  if (n > 8100000) nrep = 100
  cpu_MHz = 1000000000.0
  type='load'
  n = n/4
!!!
!!! MCK
   off1 =   noff +   1024 + 8*1024
   off2 = 2*noff + 2*1024 + 16*1024
   off3 = 3*noff + 3*1024 + 24*1024
!
! Start memory bandwidth test
!
    do irep=1,nrep
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
      end do
    end do

#if MODE==PARALLEL
    call MPI_BARRIER(MPI_COMM_WORLD,ierr)
#endif

    time1 = second()
    do irep=1,nrep
      do i=1,n-3,4
        s1 = s1 + x(i)*x(i+off1)
        s2 = s2 + x(i+1)*x(i+1+off1)
        s3 = s3 + x(i+2)*x(i+2+off1)
        s4 = s4 + x(i+3)*x(i+3+off1)
        s5 = s5 + x(i+off2)*x(i+off3)
        s6 = s6 + x(i+1+off2)*x(i+1+off3)
        s7 = s7 + x(i+2+off2)*x(i+2+off3)
        s8 = s8 + x(i+3+off2)*x(i+3+off3)
      end do
    end do
    time2 = second()
    t = time2 - time1
#if MODE==PARALLEL
!
! Do a reduce to see what was the max time by any one PE
!
  CALL MPI_ALLREDUCE(t,totalt,1,MPI_DOUBLE_PRECISION,&
        &         MPI_MAX,MPI_COMM_WORLD,ierr)
  t = totalt
#elif MODE==SERIAL
  totalt=t
  iam = 0
#endif

    y(1)=y(1)+s1 +s2 +s3 +s4+s5+s6+s7+s8

!
! print out results
!
    cyc = t*cpu_MHz/n/nrep/ns
    mb = (float(nrep)*n*8*ns/t)/1E6
!
  n = n*4
  return
  end subroutine load_mck
#endif

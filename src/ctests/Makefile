OMP	= zero_omp
SMP	= zero_smp
SHMEM	= zero_shmem
PTHREADS= zero_pthreads
SERIAL  = avail zero first second third fourth fifth overflow profile nineth tenth native eventname case1 case2 case3
ALL 	= $(OMP) $(SMP) $(SHMEM) $(PTHREADS) $(SERIAL)
UTILOBJS= do_loops.o test_utils.o

default:
	cd ..; $(MAKE)

omp: $(OMP)

smp: $(SMP)

pthreads: $(PTHREADS)

shmem: $(SHMEM)

serial: $(SERIAL)

zero_pthreads: zero_pthreads.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(CFLAGS) $(OPTFLAGS) zero_pthreads.c $(UTILOBJS) -L.. -lpapi_r $(LDFLAGS) -o zero_pthreads $(PTHRLIBS)

zero_smp: zero_smp.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(OPTFLAGS) zero_smp.c $(UTILOBJS) -L.. -lpapi_smp $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_shmem: zero_shmem.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(OPTFLAGS) zero_smp.c $(UTILOBJS) -L.. -lpapi_smp $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_omp: zero_omp.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(OMPCFLGS) $(CFLAGS) $(OPTFLAGS) zero_omp.c $(UTILOBJS) -L.. $(LDFLAGS) -o zero_omp $(OMPLIBS)

zero: zero.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) zero.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o zero

first: first.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) first.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o first

second: second.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) second.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o second

third: third.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) third.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o third

fourth: fourth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) fourth.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o fourth

fifth: fifth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) fifth.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o fifth

overflow: overflow.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) overflow.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o overflow

profile: profile.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) profile.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o profile

avail: avail.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) avail.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o avail

nineth: nineth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) nineth.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o nineth

tenth: tenth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) tenth.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o tenth

native: native.c do_loops.o ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) native.c do_loops.o -L.. -lpapi $(LDFLAGS) -o native

eventname: eventname.c do_loops.o ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) eventname.c do_loops.o -L.. -lpapi $(LDFLAGS) -o eventname

case1: case1.c ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) case1.c -L.. -lpapi $(LDFLAGS) -o case1

case2: case2.c ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) case2.c -L.. -lpapi $(LDFLAGS) -o case2

case3: case3.c ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) case3.c -L.. -lpapi $(LDFLAGS) -o case3

do_loops.o: do_loops.c
	$(CC) $(CFLAGS) -g -c do_loops.c

clean:
	rm -f *.o $(ALL) core *~
	rm -rf rii_files

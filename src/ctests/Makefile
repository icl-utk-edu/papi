OMP	= zero_omp
SMP	= zero_smp
SHMEM	= zero_shmem
PTHREADS= zero_pthreads
SERIAL  = clockres avail zero first second third fourth fifth overflow profile nineth tenth native eventname case1 case2
ALL 	= $(OMP) $(SMP) $(SHMEM) $(PTHREADS) $(SERIAL)
UTILOBJS= do_loops.o test_utils.o

omp: $(OMP)

smp: $(SMP)

pthreads: $(PTHREADS)

shmem: $(SHMEM)

serial: $(SERIAL)

zero_pthreads: zero_pthreads.c $(UTILOBJS) ../libpapi.a
	-$(CC_R) $(CFLAGS) $(OPTFLAGS) zero_pthreads.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o zero_pthreads -lpthread

zero_smp: zero_smp.c $(UTILOBJS) ../libpapi.a
	-$(CC_R) $(SMPCFLGS) $(CFLAGS) $(OPTFLAGS) zero_smp.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_shmem: zero_shmem.c $(UTILOBJS) ../libpapi.a
	-$(CC_R) $(SMPCFLGS) $(CFLAGS) $(OPTFLAGS) zero_shmem.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_omp: zero_omp.c $(UTILOBJS) ../libpapi.a
	-$(CC_R) $(OMPCFLGS) $(CFLAGS) $(OPTFLAGS) zero_omp.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o zero_omp $(OMPLIBS)

clockres: clockres.c ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) clockres.c ../libpapi.a $(LDFLAGS) -o clockres

zero: zero.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) zero.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o zero

first: first.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) first.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o first

second: second.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) second.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o second

third: third.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) third.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o third

fourth: fourth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) fourth.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o fourth

fifth: fifth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) fifth.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o fifth

overflow: overflow.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) overflow.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o overflow

profile: profile.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) profile.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o profile

avail: avail.c ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) avail.c ../libpapi.a $(LDFLAGS) -o avail

nineth: nineth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) nineth.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o nineth

tenth: tenth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) tenth.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o tenth

native: native.c do_loops.o ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) native.c do_loops.o ../libpapi.a $(LDFLAGS) -o native

eventname: eventname.c ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) eventname.c ../libpapi.a $(LDFLAGS) -o eventname

case1: case1.c ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) case1.c ../libpapi.a $(LDFLAGS) -o case1

case2: case2.c ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) case2.c ../libpapi.a $(LDFLAGS) -o case2

do_loops.o: do_loops.c
	$(CC) $(CFLAGS) $(NOOPT) -c do_loops.c

test_utils.o: test_utils.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c test_utils.c

clean:
	rm -f *.o $(ALL) core *~
	rm -rf rii_files

OMP	= zero_omp
SMP	= zero_smp
SHMEM	= zero_shmem
PTHREADS= profile_pthreads overflow_pthreads zero_pthreads 
SERIAL  = avail zero sprofile cost johnmay2 johnmay inherit clockres first second third fourth fifth overflow profile nineth tenth native eventname case1 case2
ALL 	= $(OMP) $(SMP) $(SHMEM) $(PTHREADS) $(SERIAL)
UTILOBJS= do_loops.o test_utils.o
INCLUDE = -I..

omp: $(OMP)

smp: $(SMP)

pthreads: $(PTHREADS)

shmem: $(SHMEM)

serial: $(SERIAL)

profile_pthreads: profile_pthreads.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) profile_pthreads.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o profile_pthreads -lpthread

overflow_pthreads: overflow_pthreads.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) overflow_pthreads.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o overflow_pthreads -lpthread

zero_pthreads: zero_pthreads.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_pthreads.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o zero_pthreads -lpthread

zero_smp: zero_smp.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_smp.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_shmem: zero_shmem.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_shmem.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o zero_shmem $(SMPLIBS)

zero_omp: zero_omp.c $(UTILOBJS) ../libpapi.a
	-$(CC_R) $(OMPCFLGS) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_omp.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o zero_omp $(OMPLIBS)

clockres: clockres.c ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) clockres.c ../libpapi.a $(LDFLAGS) -o clockres

sprofile: sprofile.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) sprofile.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o sprofile

cost: cost.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) cost.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o cost

inherit: inherit.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) inherit.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o inherit

johnmay: johnmay.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) johnmay.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o johnmay

johnmay2: johnmay2.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) johnmay2.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o johnmay2

zero: zero.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o zero

first: first.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) first.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o first

second: second.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) second.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o second

third: third.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) third.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o third

fourth: fourth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) fourth.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o fourth

fifth: fifth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) fifth.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o fifth

overflow: overflow.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) overflow.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o overflow

profile: profile.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) profile.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o profile

avail: avail.c ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) avail.c ../libpapi.a $(LDFLAGS) -o avail

nineth: nineth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) nineth.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o nineth

tenth: tenth.c $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) tenth.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o tenth

native: native.c $(UTILSOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) native.c $(UTILOBJS) ../libpapi.a $(LDFLAGS) -o native

eventname: eventname.c ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) eventname.c ../libpapi.a $(LDFLAGS) -o eventname

case1: case1.c ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) case1.c ../libpapi.a $(LDFLAGS) -o case1

case2: case2.c ../libpapi.a
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) case2.c ../libpapi.a $(LDFLAGS) -o case2

do_loops.o: do_loops.c
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) -c do_loops.c

test_utils.o: test_utils.c
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) -c test_utils.c

clean:
	rm -f *.o $(ALL) core *~
	rm -rf rii_files *.stderr *.stdout

run: 
	sh ./run_tests.sh

install: $(TARGETS)
	@if [ \"$(DESTDIR)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make DESTDIR=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(DESTDIR)\""; 
	mkdir -p $(DESTDIR)/ctests
	chmod go+rx $(DESTDIR)/ctests
	find . -perm -100 -type f -exec cp {} $(DESTDIR)/ctests \;
	chmod go+rx $(DESTDIR)/ctests/*

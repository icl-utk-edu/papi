OMP	= zero_omp
SMP	= zero_smp
SHMEM	= zero_shmem
PTHREADS= profile_pthreads overflow_pthreads zero_pthreads 
SERIAL  = sprofile eventname case1 case2 flops avail zero first second third third-inv fourth fourth-inv cost johnmay2 johnmay clockres fourth fifth nineth overflow_single_event overflow profile tenth native 
BROKEN	= inherit pernode
ALL 	= $(OMP) $(SMP) $(SHMEM) $(PTHREADS) $(SERIAL)
UTILOBJS= do_loops.o test_utils.o
INCLUDE = -I..
PAPILIB	= ../libpapi.a

omp: $(OMP)

smp: $(SMP)

pthreads: $(PTHREADS)

shmem: $(SHMEM)

serial: $(SERIAL)

profile_pthreads: profile_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) profile_pthreads.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o profile_pthreads -lpthread

overflow_pthreads: overflow_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) overflow_pthreads.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o overflow_pthreads -lpthread

zero_pthreads: zero_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_pthreads.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o zero_pthreads -lpthread

zero_smp: zero_smp.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_smp.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_shmem: zero_shmem.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_shmem.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o zero_shmem $(SMPLIBS)

zero_omp: zero_omp.c $(UTILOBJS) $(PAPILIB)
	-$(CC_R) $(OMPCFLGS) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_omp.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o zero_omp $(OMPLIBS)

clockres: clockres.c $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) clockres.c -L.. -lpapi $(LDFLAGS) -o clockres

sprofile: sprofile.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) sprofile.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o sprofile

cost: cost.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) cost.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o cost

inherit: inherit.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) inherit.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o inherit

johnmay: johnmay.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) johnmay.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o johnmay

johnmay2: johnmay2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) johnmay2.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o johnmay2

zero: zero.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o zero

first: first.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) first.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o first

second: second.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) second.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o second

flops: flops.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) flops.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o flops

third: third.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) third.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o third

third-inv: third-inv.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) third-inv.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o third-inv

fourth: fourth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) fourth.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o fourth

fourth-inv: fourth-inv.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) fourth-inv.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o fourth-inv

fifth: fifth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) fifth.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o fifth

overflow: overflow.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) overflow.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o overflow

overflow_single_event: overflow_single_event.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) overflow_single_event.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o overflow_single_event

profile: profile.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) profile.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o profile

pernode: pernode.c $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) pernode.c -L.. -lpapi $(LDFLAGS) -o pernode

avail: avail.c $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) avail.c -L.. -lpapi $(LDFLAGS) -o avail

nineth: nineth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) nineth.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o nineth

tenth: tenth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) tenth.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o tenth

native: native.c $(UTILSOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) native.c $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o native

eventname: eventname.c $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) eventname.c -L.. -lpapi $(LDFLAGS) -o eventname

case1: case1.c $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) case1.c -L.. -lpapi $(LDFLAGS) -o case1

case2: case2.c $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) case2.c -L.. -lpapi $(LDFLAGS) -o case2

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
	find . -name "*.[ch]" -type f -exec cp {} $(DESTDIR)/ctests \;

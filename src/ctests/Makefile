OMP	= zero_omp
SMP	= zero_smp
SHMEM	= zero_shmem
PTHREADS= profile_pthreads overflow_pthreads zero_pthreads clockres_pthreads
MPX	= multiplex1 multiplex2
MPXPTHR	= multiplex1_pthreads multiplex3_pthreads
MPI	= mpifirst
SERIAL  = zero zero_flip low-level high-level sprofile eventname case1 case2 calibrate \
	flops avail first second third fourth fourth-inv cost \
	johnmay2 johnmay clockres fourth fifth nineth \
	overflow_single_event overflow profile tenth native memory
BROKEN	= inherit pernode
UTILOBJS= do_loops.o test_utils.o dummy.o
ALL 	= $(MPI) $(MPX) $(MPXPTHR) $(OMP) $(SMP) $(SHMEM) $(PTHREADS) $(SERIAL) 
INCLUDE = -I..
PAPILIB	= ../libpapi.a

mpi: $(MPI)

multiplex_and_pthreads: $(MPX) $(PTHREADS) $(MPXPTHR)

multiplex: $(MPX) 

omp: $(OMP)

smp: $(SMP)

pthreads: $(PTHREADS)

shmem: $(SHMEM)

serial: $(SERIAL)

calibrate: calibrate.c dummy.o $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) calibrate.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o calibrate

multiplex1: multiplex1.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) multiplex1.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

multiplex2: multiplex2.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) multiplex2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

multiplex1_pthreads: multiplex1_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) multiplex1_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

multiplex3_pthreads: multiplex3_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) multiplex3_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

profile_pthreads: profile_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) profile_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile_pthreads -lpthread

overflow_pthreads: overflow_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) overflow_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_pthreads -lpthread

zero_pthreads: zero_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_pthreads -lpthread

zero_smp: zero_smp.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_smp.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_shmem: zero_shmem.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_shmem.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_shmem $(SMPLIBS)

zero_omp: zero_omp.c $(UTILOBJS) $(PAPILIB)
	-$(CC_R) $(OMPCFLGS) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_omp.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_omp $(OMPLIBS)

clockres: clockcore.c clockres.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) clockcore.c clockres.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres

clockres_pthreads: clockcore.c clockres_pthreads.c $(PAPILIB) $(UTILOBJS)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(NOOPT) clockcore.c clockres_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres_pthreads

sprofile: sprofile.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) sprofile.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o sprofile

cost: cost.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) cost.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o cost

inherit: inherit.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) inherit.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o inherit

johnmay: johnmay.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) johnmay.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o johnmay

johnmay2: johnmay2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) johnmay2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o johnmay2

zero: zero.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero

zero_flip: zero_flip.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) zero_flip.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_flip

first: first.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) first.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o first

mpifirst: mpifirst.c $(UTILOBJS) $(PAPILIB)
	$(MPICC) $(MPFLAGS) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) first.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o mpifirst

first-twice: first-twice.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) first-twice.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o first-twice

second: second.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) second.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o second

flops: flops.c dummy.o $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) flops.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o flops

third: third.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) third.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o third

fourth: fourth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) fourth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fourth

fourth-inv: fourth-inv.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) fourth-inv.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fourth-inv

fifth: fifth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) fifth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fifth

overflow: overflow.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow

overflow_single_event: overflow_single_event.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) overflow_single_event.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_single_event

profile: profile.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) profile.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile

pernode: pernode.c $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) pernode.c $(PAPILIB) $(LDFLAGS) -o pernode

avail: avail.c $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) avail.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o avail

nineth: nineth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) nineth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o nineth

tenth: tenth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) tenth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o tenth

native: native.c $(UTILSOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) native.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o native

memory: memory.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(OPTFLAGS) memory.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o memory

eventname: eventname.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) eventname.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o eventname

case1: case1.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) case1.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.c $(PAPILIB)  $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) case2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case2

low-level: low-level.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) low-level.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o low-level

high-level: high-level.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) high-level.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o high-level

dummy.o: dummy.c
	$(CC) $(CFLAGS) $(INCLUDE) $(NOOPT) -c dummy.c

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

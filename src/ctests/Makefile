OMP	= zero_omp
SMP	= zero_smp
SHMEM	= zero_shmem
PTHREADS= profile_pthreads overflow_pthreads zero_pthreads clockres_pthreads overflow3_pthreads
MPX	= multiplex1 multiplex2 mendes-alt
MPXPTHR	= multiplex1_pthreads multiplex3_pthreads
MPI	= mpifirst
SERIAL  = all_events derived high-level2 describe mem_info zero zero_flip \
	low-level high-level sprofile eventname case1 case2 calibrate \
	flops avail first second third fourth fourth-inv cost \
	johnmay2 johnmay clockres fourth fifth nineth \
	overflow_single_event overflow profile tenth native memory \
	timer_overflow overflow2 matrix-hl
BROKEN	= inherit pernode
UTILOBJS= do_loops.o test_utils.o dummy.o
ALL 	= $(MPI) $(MPX) $(MPXPTHR) $(OMP) $(SMP) $(SHMEM) $(PTHREADS) $(SERIAL) 
INCLUDE = -I..
PAPILIB	= ../$(LIBRARY)

mpi: $(MPI)

multiplex_and_pthreads: $(MPXPTHR) $(MPX) $(PTHREADS)

multiplex: $(MPX) 

omp: $(OMP)

smp: $(SMP)

pthreads: $(PTHREADS)

shmem: $(SHMEM)

serial: $(SERIAL)

calibrate: calibrate.c dummy.o $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) calibrate.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o calibrate

timer_overflow: timer_overflow.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) timer_overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

mendes-alt: mendes-alt.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -DMULTIPLEX mendes-alt.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

multiplex1: multiplex1.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) multiplex1.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

multiplex2: multiplex2.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) multiplex2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

multiplex1_pthreads: multiplex1_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) multiplex1_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

multiplex3_pthreads: multiplex3_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) multiplex3_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

overflow3_pthreads: overflow3_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow3_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

profile_pthreads: profile_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) profile_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile_pthreads -lpthread

overflow_pthreads: overflow_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_pthreads -lpthread

zero_pthreads: zero_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_pthreads -lpthread

zero_smp: zero_smp.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_smp.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_shmem: zero_shmem.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_shmem.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_shmem $(SMPLIBS)

zero_omp: zero_omp.c $(UTILOBJS) $(PAPILIB)
	-$(CC_R) $(OMPCFLGS) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_omp.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_omp $(OMPLIBS)

clockres: clockcore.c clockres.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) clockcore.c clockres.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres

clockres_pthreads: clockcore.c clockres_pthreads.c $(PAPILIB) $(UTILOBJS)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) clockcore.c clockres_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres_pthreads -lpthread

sprofile: sprofile.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) sprofile.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o sprofile

cost: cost.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) cost.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o cost

inherit: inherit.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) inherit.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o inherit

johnmay: johnmay.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) johnmay.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o johnmay

johnmay2: johnmay2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) johnmay2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o johnmay2

describe: describe.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) describe.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o describe

derived: derived.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) derived.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o derived

zero: zero.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero

zero_flip: zero_flip.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_flip.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_flip

first: first.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) first.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o first

mpifirst: mpifirst.c $(UTILOBJS) $(PAPILIB)
	$(MPICC) $(MPFLAGS) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) first.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o mpifirst

first-twice: first-twice.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) first-twice.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o first-twice

second: second.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) second.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o second

flops: flops.c dummy.o $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) flops.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o flops

third: third.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) third.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o third

fourth: fourth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) fourth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fourth

fourth-inv: fourth-inv.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) fourth-inv.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fourth-inv

fifth: fifth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) fifth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fifth

overflow: overflow.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow

overflow2: overflow2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow2

overflow_single_event: overflow_single_event.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow_single_event.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_single_event

profile: profile.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) profile.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile

pernode: pernode.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) pernode.c $(PAPILIB) $(LDFLAGS) -o pernode

mem_info: mem_info.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) mem_info.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o mem_info

all_events: all_events.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) all_events.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o all_events

avail: avail.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) avail.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o avail

nineth: nineth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) nineth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o nineth

tenth: tenth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) tenth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o tenth

native: native.c $(UTILSOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) native.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o native

memory: memory.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) memory.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o memory

eventname: eventname.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) eventname.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o eventname

case1: case1.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) case1.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.c $(PAPILIB)  $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) case2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case2

low-level: low-level.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) low-level.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o low-level

matrix-hl: matrix-hl.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) matrix-hl.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o matrix-hl

high-level: high-level.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) high-level.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o high-level

high-level2: high-level2.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) high-level2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o high-level2

dummy.o: dummy.c
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c dummy.c

do_loops.o: do_loops.c
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c do_loops.c

test_utils.o: test_utils.c
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c test_utils.c

clean:
	rm -f *.o *.stderr *.stdout core *~ $(ALL)

run: 
	sh ./run_tests.sh

install: $(TARGETS)
	@if [ \"$(DESTDIR)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make DESTDIR=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(DESTDIR)\""; 
	-mkdir -p $(DESTDIR)/ctests
	chmod go+rx $(DESTDIR)/ctests
	find . -perm -100 -type f -exec cp {} $(DESTDIR)/ctests \;
	chmod go+rx $(DESTDIR)/ctests/*
	find . -name "*.[ch]" -type f -exec cp {} $(DESTDIR)/ctests \;

OMP	= zero_omp
SMP	= zero_smp
SHMEM	= zero_shmem
PTHREADS= profile_pthreads overflow_pthreads zero_pthreads clockres_pthreads overflow3_pthreads locks_pthreads 
MPX	= multiplex1 multiplex2 mendes-alt sdsc-mpx sdsc2-mpx sdsc4-mpx
MPXPTHR	= multiplex1_pthreads multiplex3_pthreads
MPI	= mpifirst
SHARED  = shlib
SERIAL  = realtime first exeinfo zero zero_fork avail branches dmem_info all_events derived high-level2 describe mem_info memory  \
	zero_flip low-level high-level eventname case1 case2 \
	calibrate flops second cost johnmay2 clockres \
	overflow overflow_single_event overflow_twoevents timer_overflow overflow2 \
	matrix-hl tenth \
	profile  sprofile profile_twoevents earprofile\
	ipc native native_avail overflow_index command_line
BROKEN	= inherit pernode
UTILOBJS= do_loops.o test_utils.o dummy.o
TESTS 	= $(PTHREADS) $(SERIAL) $(MPI) $(MPX) $(MPXPTHR) $(OMP) $(SMP) $(SHMEM) $(PTHREADS) $(SHARED)
INCLUDE = -I.. -I/usr/include 
PAPILIB	= ../$(LIBRARY)

mpi: $(MPI)

shared: $(SHARED)

multiplex_and_pthreads: $(MPXPTHR) $(MPX) $(PTHREADS)

multiplex: $(MPX) 

omp: $(OMP)

smp: $(SMP)

pthreads: $(PTHREADS)

shmem: $(SHMEM)

serial: $(SERIAL)

sdsc2: sdsc2.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) sdsc.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -lm -o $@

sdsc2-mpx: sdsc2.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) -DMPX $(INCLUDE) $(TOPTFLAGS) sdsc2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -lm -o $@

branches: branches.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) branches.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -lm -o $@

sdsc2-mpx-noreset: sdsc2.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) -DMPX -DSTARTSTOP $(INCLUDE) $(TOPTFLAGS) sdsc.c $(UTILOBJS) $(PAPILIB) -lm $(LDFLAGS) -o $@

sdsc-mpx: sdsc.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) -DMPX $(INCLUDE) $(TOPTFLAGS) sdsc.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@

sdsc4-mpx: sdsc4.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) -DMPX $(INCLUDE) $(TOPTFLAGS) sdsc4.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -lm -o $@

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

locks_pthreads: locks_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) locks_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o locks_pthreads -lpthread

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
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) clockcore.c clockres.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres -lm

clockres_pthreads: clockcore.c clockres_pthreads.c $(PAPILIB) $(UTILOBJS)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) clockcore.c clockres_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres_pthreads -lpthread -lm

sprofile: sprofile.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) sprofile.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o sprofile

cost: cost.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) cost.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o cost -lm

inherit: inherit.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) inherit.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o inherit

johnmay2: johnmay2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) johnmay2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o johnmay2

describe: describe.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) describe.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o describe

derived: derived.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) derived.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o derived

zero: zero.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero

zero_fork: zero_fork.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_fork.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_fork

try: try.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) try.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o try

zero_flip: zero_flip.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_flip.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_flip

realtime: realtime.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) realtime.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o realtime

command_line: command_line.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) command_line.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o command_line

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

ipc: ipc.c dummy.o $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) ipc.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o ipc

overflow: overflow.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow

overflow_twoevents: overflow_twoevents.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow_twoevents.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_twoevents

overflow_index: overflow_index.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow_index.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_index

overflow2: overflow2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow2

overflow_single_event: overflow_single_event.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow_single_event.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_single_event

profile: profile.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) profile.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile

profile_twoevents: profile_twoevents.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) profile_twoevents.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile_twoevents

earprofile: earprofile.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) earprofile.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o earprofile

pernode: pernode.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) pernode.c $(PAPILIB) $(LDFLAGS) -o pernode

dmem_info: dmem_info.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) dmem_info.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o dmem_info

mem_info: mem_info.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) mem_info.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o mem_info

all_events: all_events.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) all_events.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o all_events

avail: avail.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) avail.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o avail

memory: memory.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) memory.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o memory

native_avail: native_avail.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) native_avail.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o native_avail

tenth: tenth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) tenth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o tenth

native: native.c $(UTILSOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) native.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o native

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

shlib: shlib.c $(PAPILIB) $(UTILOBJS)
	-$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) shlib.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o shlib -ldl
	if [ ! -x ./shlib ]; then $(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) shlib.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o shlib; fi

exeinfo: exeinfo.c $(PAPILIB) $(UTILOBJS)
	-$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) exeinfo.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o exeinfo 

test_utils.o: test_utils.c papi_test.h test_utils.h
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c test_utils.c

do_loops.o: do_loops.c papi_test.h test_utils.h
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c do_loops.c

dummy.o: dummy.c
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c dummy.c

clean:
	rm -f *.o *.stderr *.stdout core *~ $(TESTS)

install: $(TARGETS)
	@if [ \"$(PREFIX)\" = \"\" ]; then echo "You must specify a destination directory on the make line"; echo "For example: make PREFIX=/usr/local"; exit 1; fi 
	@echo "Root of destination directory is: \"$(PREFIX)\""; 
	-mkdir -p $(PREFIX)/share/papi/ctests
	-chmod go+rx $(PREFIX)/share/papi/ctests
	-find . -perm -100 -type f -exec cp {} $(PREFIX)/share/papi/ctests \;
	-chmod go+rx $(PREFIX)/share/papi/ctests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(PREFIX)/share/papi/ctests \;

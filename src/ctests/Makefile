OMP	= zero_omp
SMP	= zero_smp
SHMEM	= zero_shmem
PTHREADS= thrspecific profile_pthreads overflow_pthreads zero_pthreads unregister_pthreads clockres_pthreads overflow3_pthreads locks_pthreads 
MPX	= multiplex1 multiplex2 mendes-alt sdsc-mpx sdsc2-mpx sdsc4-mpx
MPXPTHR	= multiplex1_pthreads multiplex3_pthreads kufrin
MPI	= mpifirst
SHARED  = shlib
SERIAL  = multiattach zero_attach reset subinfo realtime first exeinfo zero zero_fork branches dmem_info all_native_events all_events derived high-level2 describe memory  \
	zero_flip low-level high-level eventname case1 case2 \
	calibrate flops second johnmay2 \
	overflow overflow_force_software overflow_single_event \
	overflow_twoevents timer_overflow overflow2 \
	matrix-hl tenth encode encode2 \
	profile sprofile profile_twoevents byte_profile\
	ipc native overflow_index overflow_one_and_read
P4_TEST	= p4_lst_ins
ACPI	= acpi 
EAR	= earprofile
RANGE	= data_range
BROKEN	= inherit pernode
UTILOBJS= do_loops.o test_utils.o dummy.o
TESTS	= $(PTHREADS) $(SERIAL) $(MPI) $(MPX) $(MPXPTHR) $(OMP) $(SMP) $(SHMEM)\
	$(SHARED) $(EAR) $(RANGE) $(P4_TEST) $(ACPI)
INCLUDE = -I.. -I/usr/include 
PAPILIB	= ../$(LIBRARY)

p4: $(P4_TEST)

acpi_test: $(ACPI)

ear: $(EAR)

range: $(RANGE)

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

data_range: data_range.c dummy.o $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) data_range.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o data_range

p4_lst_ins: p4_lst_ins.c dummy.o $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) p4_lst_ins.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o p4_lst_ins

acpi: acpi.c dummy.o $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) acpi.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o acpi

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

kufrin: kufrin.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) kufrin.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

multiplex3_pthreads: multiplex3_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) multiplex3_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

overflow3_pthreads: overflow3_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow3_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

thrspecific: thrspecific.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) thrspecific.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o thrspecific -lpthread

profile_pthreads: profile_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) profile_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile_pthreads -lpthread

locks_pthreads: locks_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) locks_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o locks_pthreads -lpthread

overflow_pthreads: overflow_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_pthreads -lpthread

zero_pthreads: zero_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_pthreads -lpthread

unregister_pthreads: zero_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -DTEST_UNREGISTER zero_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

zero_smp: zero_smp.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_smp.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_shmem: zero_shmem.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_shmem.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_shmem $(SMPLIBS)

zero_omp: zero_omp.c $(UTILOBJS) $(PAPILIB)
	-$(CC_R) $(OMPCFLGS) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_omp.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_omp $(OMPLIBS)

val_omp: val_omp.c $(UTILOBJS) $(PAPILIB)
	-$(CC_R) $(OMPCFLGS) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) val_omp.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o val_omp $(OMPLIBS)

clockres_pthreads: clockcore.c clockres_pthreads.c $(PAPILIB) $(UTILOBJS)
	$(CC_R) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) clockcore.c clockres_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres_pthreads -lpthread -lm

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

overflow_one_and_read: overflow_one_and_read.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow_one_and_read.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_one_and_read

overflow_index: overflow_index.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow_index.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_index

overflow2: overflow2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow2

overflow_single_event: overflow_single_event.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow_single_event.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_single_event

overflow_force_software: overflow_force_software.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) overflow_force_software.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_force_software

sprofile: sprofile.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) sprofile.c prof_utils.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o sprofile

profile: profile.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) profile.c prof_utils.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile

profile_twoevents: profile_twoevents.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) profile_twoevents.c prof_utils.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile_twoevents

earprofile: earprofile.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) earprofile.c $(UTILOBJS) prof_utils.o $(PAPILIB) $(LDFLAGS) -o earprofile

byte_profile: byte_profile.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) byte_profile.c prof_utils.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o byte_profile

pernode: pernode.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) pernode.c $(PAPILIB) $(LDFLAGS) -o pernode

dmem_info: dmem_info.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) dmem_info.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o dmem_info

all_events: all_events.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) all_events.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o all_events

all_native_events: all_native_events.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) all_native_events.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o all_native_events

memory: memory.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) memory.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o memory

tenth: tenth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) tenth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o tenth

native: native.c $(UTILSOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) native.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o native

eventname: eventname.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) eventname.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o eventname

encode: encode.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) encode.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o encode

encode2: encode2.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) encode2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o encode2

case1: case1.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) case1.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.c $(PAPILIB)  $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) case2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case2

low-level: low-level.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) low-level.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o low-level

matrix-hl: matrix-hl.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -O2 matrix-hl.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o matrix-hl

high-level: high-level.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) high-level.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o high-level

high-level2: high-level2.c $(PAPILIB) $(UTILOBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) high-level2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o high-level2

shlib: shlib.c $(PAPILIB) $(UTILOBJS)
	-$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) shlib.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o shlib -ldl
	if [ ! -x ./shlib ]; then $(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) shlib.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o shlib; fi

exeinfo: exeinfo.c $(PAPILIB) $(UTILOBJS)
	-$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) exeinfo.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o exeinfo 

subinfo: subinfo.c $(PAPILIB) $(UTILOBJS)
	-$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) subinfo.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o subinfo 

zero_attach: zero_attach.c $(PAPILIB) $(UTILOBJS)
	-$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) zero_attach.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_attach 

multiattach: multiattach.c $(PAPILIB) $(UTILOBJS)
	-$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) multiattach.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o multiattach 

reset: reset.c $(PAPILIB) $(UTILOBJS)
	-$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) reset.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o reset 

prof_utils.o: prof_utils.c papi_test.h prof_utils.h
	$(CC) $(CFLAGS) $(INCLUDE) $(TOPTFLAGS) -c prof_utils.c

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

# File: ctests/Makefile
# CVS:  $Id$
INCLUDE = -I.. -I. -I../testlib
PAPILIB = ../libpapi.a
CC	= gcc
CC_R	= $(CC) -pthread
CFLAGS	= -g -O -Wall

OMP	= zero_omp omptough
SMP	= zero_smp
SHMEM	= zero_shmem
PTHREADS= pthrtough pthrtough2 thrspecific profile_pthreads overflow_pthreads zero_pthreads clockres_pthreads overflow3_pthreads locks_pthreads krentel_pthreads
MPX	= max_multiplex multiplex1 multiplex2 mendes-alt sdsc-mpx sdsc2-mpx sdsc4-mpx
MPXPTHR	= multiplex1_pthreads multiplex3_pthreads kufrin
MPI	= mpifirst
SHARED  = shlib
SERIAL  = inherit hwinfo code2name reset subinfo realtime virttime first exeinfo zero \
	zero_named branches dmem_info all_native_events all_events derived high-level2 describe memory  \
	zero_flip low-level high-level eventname case1 case2 \
	calibrate flops second johnmay2 matrix-hl tenth ipc nmi_watchdog
FORKEXEC  = fork fork2 exec exec2 forkexec forkexec2 forkexec3 forkexec4 fork_overflow exec_overflow \
	child_overflow system_child_overflow system_overflow burn zero_fork
OVERFLOW  = fork_overflow exec_overflow child_overflow system_child_overflow system_overflow burn \
	overflow overflow_force_software overflow_single_event overflow_twoevents timer_overflow overflow2 \
	overflow_index overflow_one_and_read overflow_allcounters
PROFILE  = profile profile_force_software sprofile profile_twoevents byte_profile
ATTACH	= multiattach multiattach2 zero_attach attach3 attach2 attach_target
P4_TEST	= p4_lst_ins
ACPI	= acpi 
EAR	= earprofile
RANGE	= data_range
BROKEN	= pernode
API = api
ALL	= $(PTHREADS) $(SERIAL) $(FORKEXEC) $(OVERFLOW) $(PROFILE) $(MPI) $(MPX) $(MPXPTHR) $(OMP) $(SMP) $(SHMEM)\
	$(SHARED) $(EAR) $(RANGE) $(P4_TEST) $(ACPI) $(ATTACH) $(API)
UTILOBJS= ../testlib/do_loops.o ../testlib/test_utils.o ../testlib/dummy.o

default all ctests ctest: $(CTEST_TARGETS)

attach: $(ATTACH)

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

forkexec_tests: $(FORKEXEC)

overflow_tests: $(OVERFLOW)

profile_tests: $(PROFILE)

papi_api: $(API)

api: api.c $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) api.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@

sdsc2: sdsc2.c $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) sdsc.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -lm -o $@

sdsc2-mpx: sdsc2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) -DMPX $(TOPTFLAGS) sdsc2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -lm -o $@

branches: branches.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) branches.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -lm -o $@

sdsc2-mpx-noreset: sdsc2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) -DMPX -DSTARTSTOP $(TOPTFLAGS) sdsc.c $(UTILOBJS) $(PAPILIB) -lm $(LDFLAGS) -o $@

sdsc-mpx: sdsc.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) -DMPX $(TOPTFLAGS) sdsc.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@

sdsc4-mpx: sdsc4.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) -DMPX $(TOPTFLAGS) sdsc4.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -lm -o $@

calibrate: calibrate.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) calibrate.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o calibrate

data_range: data_range.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) data_range.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o data_range

p4_lst_ins: p4_lst_ins.c $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) p4_lst_ins.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o p4_lst_ins

acpi: acpi.c dummy.o $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) acpi.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o acpi

timer_overflow: timer_overflow.c $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) timer_overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

mendes-alt: mendes-alt.c $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -DMULTIPLEX mendes-alt.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

max_multiplex: max_multiplex.c $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) max_multiplex.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

multiplex1: multiplex1.c $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) multiplex1.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

multiplex2: multiplex2.c $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) multiplex2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ 

multiplex1_pthreads: multiplex1_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) multiplex1_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

kufrin: kufrin.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) kufrin.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

multiplex3_pthreads: multiplex3_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) multiplex3_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

overflow3_pthreads: overflow3_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow3_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o $@ -lpthread

thrspecific: thrspecific.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) thrspecific.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o thrspecific -lpthread

pthrtough: pthrtough.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) pthrtough.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o pthrtough -lpthread

pthrtough2: pthrtough2.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) pthrtough2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o pthrtough2 -lpthread

profile_pthreads: profile_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) profile_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile_pthreads -lpthread

locks_pthreads: locks_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) locks_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o locks_pthreads -lpthread

krentel_pthreads: krentel_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) krentel_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o krentel_pthreads -lpthread

overflow_pthreads: overflow_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_pthreads -lpthread

zero_pthreads: zero_pthreads.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) zero_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_pthreads -lpthread

zero_smp: zero_smp.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(SMPCFLGS) $(CFLAGS) $(TOPTFLAGS) zero_smp.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_shmem: zero_shmem.c $(UTILOBJS) $(PAPILIB)
	$(CC_R) $(INCLUDE) $(SMPCFLGS) $(CFLAGS) $(TOPTFLAGS) zero_shmem.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_shmem $(SMPLIBS)

zero_omp: zero_omp.c $(UTILOBJS) $(PAPILIB)
	-$(CC_R) $(INCLUDE) $(OMPCFLGS) $(CFLAGS) $(TOPTFLAGS) zero_omp.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_omp $(OMPLIBS)

omptough: omptough.c $(UTILOBJS) $(PAPILIB)
	-$(CC_R) $(INCLUDE) $(OMPCFLGS) $(CFLAGS) $(TOPTFLAGS) omptough.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o omptough $(OMPLIBS)

val_omp: val_omp.c $(UTILOBJS) $(PAPILIB)
	-$(CC_R) $(INCLUDE) $(OMPCFLGS) $(CFLAGS) $(TOPTFLAGS) val_omp.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o val_omp $(OMPLIBS)

clockres_pthreads: clockcore.c clockres_pthreads.c $(UTILOBJS) $(PAPILIB) 
	$(CC_R) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) clockcore.c clockres_pthreads.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o clockres_pthreads -lpthread -lm

inherit: inherit.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) inherit.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o inherit

johnmay2: johnmay2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) johnmay2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o johnmay2

describe: describe.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) describe.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o describe

derived: derived.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) derived.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o derived

zero: zero.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) zero.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero

zero_named: zero_named.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) zero_named.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_named

nmi_watchdog: nmi_watchdog.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) nmi_watchdog.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o nmi_watchdog

zero_fork: zero_fork.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) zero_fork.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_fork

try: try.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) try.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o try

zero_flip: zero_flip.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) zero_flip.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_flip

realtime: realtime.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) realtime.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o realtime

virttime: virttime.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) virttime.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o virttime

first: first.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) first.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o first

mpifirst: mpifirst.c $(UTILOBJS) $(PAPILIB)
	$(MPICC) $(INCLUDE) $(MPFLAGS) $(CFLAGS) $(TOPTFLAGS) first.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o mpifirst

first-twice: first-twice.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) first-twice.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o first-twice

second: second.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) second.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o second

flops: flops.c $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) flops.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o flops

ipc: ipc.c $(UTILOBJS) $(PAPILIB) 
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) ipc.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o ipc

overflow: overflow.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow

overflow_allcounters: overflow_allcounters.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow_allcounters.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_allcounters

overflow_twoevents: overflow_twoevents.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow_twoevents.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_twoevents

overflow_one_and_read: overflow_one_and_read.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow_one_and_read.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_one_and_read

overflow_index: overflow_index.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow_index.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_index

overflow_values: overflow_values.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow_values.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_values

overflow2: overflow2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow2

overflow_single_event: overflow_single_event.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow_single_event.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_single_event

overflow_force_software: overflow_force_software.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) overflow_force_software.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o overflow_force_software

sprofile: sprofile.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) sprofile.c prof_utils.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o sprofile

profile: profile.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) profile.c prof_utils.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile

profile_force_software: profile.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -DSWPROFILE profile.c prof_utils.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile_force_software

profile_twoevents: profile_twoevents.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) profile_twoevents.c prof_utils.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o profile_twoevents

earprofile: earprofile.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) earprofile.c $(UTILOBJS) prof_utils.o $(PAPILIB) $(LDFLAGS) -o earprofile

byte_profile: byte_profile.c $(UTILOBJS) prof_utils.o $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) byte_profile.c prof_utils.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o byte_profile

pernode: pernode.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) pernode.c $(LDFLAGS) -o pernode

dmem_info: dmem_info.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) dmem_info.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o dmem_info

all_events: all_events.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) all_events.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o all_events

all_native_events: all_native_events.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) all_native_events.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o all_native_events

memory: memory.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) memory.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o memory

tenth: tenth.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) tenth.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o tenth

eventname: eventname.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) eventname.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o eventname

case1: case1.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) case1.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case1

case2: case2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) case2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o case2

low-level: low-level.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) low-level.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o low-level

matrix-hl: matrix-hl.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS)  matrix-hl.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o matrix-hl

high-level: high-level.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) high-level.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o high-level

high-level2: high-level2.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) high-level2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o high-level2

shlib: shlib.c $(UTILOBJS) $(PAPILIB)
		$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) shlib.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o shlib $(LDL)

exeinfo: exeinfo.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) exeinfo.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o exeinfo 

subinfo: subinfo.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) subinfo.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o subinfo 

hwinfo: hwinfo.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) hwinfo.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o hwinfo

code2name: code2name.c $(UTILOBJS) $(PAPILIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) code2name.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o code2name

attach_target: ../testlib/do_loops.c
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -DDUMMY_DRIVER ../testlib/do_loops.c ../testlib/dummy.c -o attach_target 

zero_attach: zero_attach.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) zero_attach.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o zero_attach 

multiattach: multiattach.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) multiattach.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o multiattach 

multiattach2: multiattach2.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) multiattach2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o multiattach2 

attach3: attach3.c attach_target $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) attach3.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o attach3 

attach2: attach2.c attach_target $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) attach2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o attach2 

reset: reset.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) reset.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o reset 

fork_overflow: fork_exec_overflow.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) fork_exec_overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fork_overflow

exec_overflow: fork_exec_overflow.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -DPEXEC fork_exec_overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o exec_overflow

child_overflow: fork_exec_overflow.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -DPCHILD fork_exec_overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o child_overflow

system_child_overflow: fork_exec_overflow.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -DSYSTEM fork_exec_overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o system_child_overflow

system_overflow: fork_exec_overflow.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -DSYSTEM2 fork_exec_overflow.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o system_overflow

burn: burn.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) burn.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o burn

fork: fork.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) fork.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fork 

exec: exec.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) exec.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o exec 

exec2: exec2.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) exec2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o exec2 

fork2: fork2.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) fork2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o fork2 

forkexec: forkexec.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) forkexec.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o forkexec 

forkexec2: forkexec2.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) forkexec2.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o forkexec2 

forkexec3: forkexec3.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) forkexec3.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o forkexec3 

forkexec4: forkexec4.c $(UTILOBJS) $(PAPILIB)
	-$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) forkexec4.c $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -o forkexec4 

../testlib/test_utils.o:	../testlib/test_utils.c ../testlib/papi_test.h ../testlib/test_utils.h
	cd ../testlib && $(MAKE)

../testlib/do_loops.o:	../testlib/do_loops.c ../testlib/papi_test.h ../testlib/test_utils.h
	cd ../testlib && $(MAKE)

../testlib/dummy.o:	../testlib/dummy.c
	cd ../testlib && $(MAKE)

prof_utils.o: prof_utils.c ../testlib/papi_test.h prof_utils.h
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c prof_utils.c

clean:
	rm -f *.o *.stderr *.stdout core *~ $(ALL) unregister_pthreads

install: $(CTEST_TARGETS)
	@echo "C tests (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/ctests
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/ctests
	-find . -perm -100 -type f -exec cp {} $(DATADIR)/ctests \;
	-chmod go+rx $(DATADIR)/ctests/*
	-find . -name "*.[ch]" -type f -exec cp {} $(DATADIR)/ctests \;

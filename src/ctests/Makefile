OMP	= zero_omp
SMP	= zero_smp
PTHREADS= zero_pthreads
SERIAL  = avail zero first second third fourth fifth overflow profile nineth tenth native eventname
ALL 	= $(OMP) $(SMP) $(PTHREADS) $(SERIAL)
UTILOBJS= do_loops.o test_utils.o

default:
	cd ..; $(MAKE)
omp: $(OMP)

smp: $(SMP)

pthreads: $(PTHREADS)

serial: $(SERIAL)

zero_pthreads: zero_pthreads.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(CFLAGS) $(OPTFLAGS) zero_pthreads.c $(UTILOBJS) -L.. $(LDFLAGS) -o zero_pthreads $(PTHRLIBS)

zero_smp: zero_smp.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(SMPCFLGS) $(CFLAGS) $(OPTFLAGS) zero_smp.c $(UTILOBJS) -L.. $(LDFLAGS) -o zero_smp $(SMPLIBS)

zero_omp: zero_omp.c $(UTILOBJS) ../libpapi.a
	$(CC_R) $(OMPCFLGS) $(CFLAGS) $(OPTFLAGS) zero_omp.c $(UTILOBJS) -L.. $(LDFLAGS) -o zero_omp $(OMPLIBS)

zero: zero.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) zero.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o zero

first: first.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) first.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o first

second: second.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) second.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o second

third: third.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) third.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o third

fourth: fourth.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) fourth.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o fourth

fifth: fifth.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) fifth.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o fifth

overflow: overflow.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) overflow.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o overflow

profile: profile.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) profile.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o profile

avail: avail.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) avail.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o avail

nineth: nineth.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) nineth.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o nineth

tenth: tenth.o $(UTILOBJS) ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) tenth.o $(UTILOBJS) -L.. -lpapi $(LDFLAGS) -o tenth

native: native.o do_loops.o ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) native.o do_loops.o -L.. -lpapi $(LDFLAGS) -o native

eventname: eventname.o do_loops.o ../libpapi.a
	$(CC) $(CFLAGS) $(OPTFLAGS) eventname.o do_loops.o -L.. -lpapi $(LDFLAGS) -o eventname

do_loops.o: do_loops.c
	$(CC) $(CFLAGS) -g -c do_loops.c

clean:
	rm -f *.o $(ALL) core *~

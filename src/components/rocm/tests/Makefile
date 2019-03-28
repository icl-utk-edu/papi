NAME=rocm
include ../../Makefile_comp_tests.target
INCLUDE += -I/opt/rocm/hip/include

%.o:%.c
	@echo "INCLUDE=" $(INCLUDE)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

%.o:%.cpp
	@echo "INCLUDE=" $(INCLUDE)
	@echo "CFLAGS=" $(CFLAGS)
	g++ $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

TESTS = 

rocm_tests: $(TESTS)

rocm_command_line: rocm_command_line.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o rocm_command_line rocm_command_line.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

rocm_all: rocm_all.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o rocm_all rocm_all.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) 

clean:
	rm -f $(TESTS) *.o

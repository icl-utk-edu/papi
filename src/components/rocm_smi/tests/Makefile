NAME=rocm_smi
include ../../Makefile_comp_tests.target

TESTS = 

rocm_smi_tests: $(TESTS)

# We have no standardized tests yet. the below are examples of what one might look like;
# but the "-fverbose-asm -Wa,-adhln=testPCP.s" are not usually included; those are for 
# advanced debugging. So is "-Xlinker -Map=testPCP_link.map"; those can be left off.
#testPCP.o:	testPCP.c
#	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c testPCP.c -o testPCP.o -fverbose-asm -Wa,-adhln=testPCP.s

#testPCP: testPCP.o $(UTILOBJS) $(PAPILIB)
#	$(CC) $(INCLUDE) -o testPCP testPCP.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS) -Xlinker -Map=testPCP_link.map

clean:
	rm -f $(TESTS) *.o *~

#define NUM_FLOPS 10000000

/* Masks to select operations for add_test_events() and remove_test_events()
	Mask value tells us what events to select.
*/
#define MASK_BR_PRC  0x2000
#define MASK_BR_MSP  0x1000
#define MASK_BR_CN   0x0800
#define MASK_L2_TCH  0x0400
#define MASK_L2_TCA  0x0200
#define MASK_L2_TCM  0x0100
#define MASK_L1_DCM  0x0040
#define MASK_L1_ICM  0x0020
#define MASK_L1_TCM  0x0010
#define MASK_FLOPS   0x0008
#define MASK_FP_INS  0x0004
#define MASK_TOT_INS 0x0002
#define MASK_TOT_CYC 0x0001

void *get_overflow_address(void *context);
void free_test_space(long_long **values, int num_tests);
long_long **allocate_test_space(int num_tests, int num_events);
int add_test_events(int *number, int *mask);
int add_test_events_r(int *number, int *mask, void *handle);
int remove_test_events(int *EventSet, int mask);
void do_flops(int n);
void do_reads(int n);
void do_both(int n);
void do_l1misses(int n);
char *stringify_domain(int domain);
char *stringify_granularity(int granularity);
void tests_quiet(int argc, char **argv);
void test_pass(char *file, long_long **values, int num_tests);
void test_fail(char *file, int line, char *call, int retval);
void test_skip(char *file, int line, char *call, int retval);


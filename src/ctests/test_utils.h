#define NUM_FLOPS 10000000
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

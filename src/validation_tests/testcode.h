#define ALL_OK              0
#define CODE_UNIMPLEMENTED -1
#define ERROR_RESULT       -2

/* instructions_testcode.c */
int instructions_million(void);
int instructions_fldcw(void);
int instructions_rep(void);

/* branches_testcode.c */
int branches_testcode(void);
int random_branches_testcode(int number, int quiet);

/* flops_testcode.c */
int flops_init_matrix(void);
float flops_matrix_matrix_multiply(void);
float flops_swapped_matrix_matrix_multiply(void);
double do_flops3( double x, int iters, int quiet );

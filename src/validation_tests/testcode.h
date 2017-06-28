#define ALL_OK              0
#define CODE_UNIMPLEMENTED -1
#define ERROR_RESULT       -2

int instructions_million(void);
int instructions_fldcw(void);
int instructions_rep(void);

int branches_testcode(void);
int random_branches_testcode(int number, int quiet);

int flops_init_matrix(void);
float flops_matrix_matrix_multiply(void);
float flops_swapped_matrix_matrix_multiply(void);

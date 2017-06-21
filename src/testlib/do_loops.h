#define NUM_FLOPS  20000000
#define L1_MISS_BUFFER_SIZE_INTS 128*1024

void do_reads( int n );
void fdo_reads( int *n );
void fdo_reads_( int *n );
void fdo_reads__( int *n );
void FDO_READS( int *n );
void _FDO_READS( int *n );
void do_flops( int n );
void fdo_flops( int *n );
void fdo_flops_( int *n );
void fdo_flops__( int *n );
void FDO_FLOPS( int *n );
void _FDO_FLOPS( int *n );
void do_misses( int n, int bytes );
void fdo_misses( int *n, int *size );
void fdo_misses_( int *n, int *size );
void fdo_misses__( int *n, int *size );
void FDO_MISSES( int *n, int *size );
void _FDO_MISSES( int *n, int *size );
void do_flush( void );
void fdo_flush( void );
void fdo_flush_( void );
void fdo_flush__( void );
void FDO_FLUSH( void );
void _FDO_FLUSH( void );
void do_l1misses( int n );
void fdo_l1misses( int *n );
void fdo_l1misses_( int *n );
void fdo_l1misses__( int *n );
void FDO_L1MISSES( int *n );
void _FDO_L1MISSES( int *n );
void do_stuff( void );
void do_stuff_( void );
void do_stuff__( void );
void DO_STUFF( void );
void _DO_STUFF( void );

void dummy( void *array );
void dummy_( void *array );
void dummy__( void *array );
void DUMMY( void *array );
void _DUMMY( void *array );
void touch_dummy( double *array, int size );

#ifndef _PAPI_MALLOC
#define _PAPI_MALLOC
#define DEBUG_FILE_LEN  20

/* If you are tracing memory, then PAPI_DEBUG_MEMORY
 *  * must be set also.
 *   */
#ifdef PAPI_NO_MEMORY_MANAGEMENT
#undef PAPI_DEBUG_MEMORY
#undef PAPI_DEBUG_MEMORY_TRACE
#else
#ifdef PAPI_DEBUG_MEMORY_TRACE
#define PAPI_DEBUG_MEMORY
#endif
#endif

typedef struct pmem {
  void *ptr;
  int size;
#ifdef PAPI_DEBUG_MEMORY
  char file[DEBUG_FILE_LEN];
  int  line;
#endif
  struct pmem *next;
  struct pmem *prev;
} pmem_t;

#ifndef IN_MEM_FILE
#ifdef PAPI_NO_MEMORY_MANAGEMENT
#define papi_malloc(a) malloc(a)
#define papi_free(a)   free(a)
#define papi_realloc(a,b) realloc(a,b)
#define papi_calloc(a,b) calloc(a,b)
#define papi_valid_free(a) ;
#define papi_strdup(a) strdup(a)
#define _papi_cleanup_all_memory() ;
#else
#define papi_malloc(a) _papi_malloc(__FILE__,__LINE__, a)
#define papi_free(a) _papi_free(__FILE__,__LINE__, a)
#define papi_realloc(a,b) _papi_realloc(__FILE__,__LINE__,a,b)
#define papi_calloc(a,b) _papi_calloc(__FILE__,__LINE__,a,b)
#define papi_valid_free(a) _papi_valid_free(__FILE__,__LINE__,a)
#define papi_strdup(a) _papi_strdup(__FILE__,__LINE__,a)
void _papi_cleanup_all_memory();
#endif
#endif

char * _papi_strdup(char *, int, const char *s);
void _papi_mem_print_info(void *ptr);
void *_papi_malloc(char *, int, int);
void _papi_free(char *, int, void *);
void _papi_valid_free(char *, int, void *);
void *_papi_realloc(char *, int, void *, int);
void *_papi_calloc(char *, int, int, int);
int check_buffer_overflow(pmem_t *);
int check_memory_for_overflow();
int papi_mem_overhead(int);

#define PAPI_MEM_LIB_OVERHEAD	1   /* PAPI Library Overhead */
#define PAPI_MEM_OVERHEAD	2   /* Memory Overhead */
#endif

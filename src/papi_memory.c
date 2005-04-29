/*
 * File:    papi_memory.c
 * CVS:     $Id$
 * Author:  Kevin London
 *          london@cs.utk.edu
 * Mods:    <Your name here>
 *          <Your email here>
 */

/* STILL NEED LOCKING FOR THREADS */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

/* If you are tracing memory, then PAPI_DEBUG_MEMORY
 * must be set also.
 */
#ifdef PAPI_DEBUG_MEMORY_TRACE
#define PAPI_DEBUG_MEMORY 1
#endif

#define END_PAD 4
#define END_PATTERN_1 0xC
#define END_PATTERN_2 0xA
#define END_PATTERN_3 0xC
#define END_PATTERN_4 0xA

/* Local global variables */
static pmem_t *mem_head=NULL;

/* Local Prototypes */
pmem_t * init_mem_ptr(void *, int, char *, int);
void remove_mem_ptr(pmem_t *);
void insert_mem_ptr(pmem_t *);


inline pmem_t * get_mem_ptr(void * ptr){
  pmem_t **tmp_ptr = (pmem_t **) (ptr - sizeof(void *));
  pmem_t *mem_ptr;

  if ( !tmp_ptr ) return(NULL);

  mem_ptr = *tmp_ptr;
  return (mem_ptr);
}

void *_papi_realloc(char *file, int line, void *ptr, int size){
  int nsize = size+sizeof(void *);
  pmem_t *mem_ptr; 
  void *nptr;
#ifdef PAPI_DEBUG_MEMORY
  char *chptr;

  nsize+=END_PAD;
  check_memory_for_overflow();
#endif

  if ( !ptr ) return(_papi_malloc(file, line, size));

  mem_ptr = get_mem_ptr(ptr);
  nptr = (pmem_t *) realloc((ptr-sizeof(void*)), nsize);

  if ( !nptr ) return(NULL);

  mem_ptr->size = size;
  mem_ptr->ptr = nptr+sizeof(void*);
#ifdef PAPI_DEBUG_MEMORY
  strncpy(mem_ptr->file, file, DEBUG_FILE_LEN);
  mem_ptr->file[DEBUG_FILE_LEN-1] = '\0';
  mem_ptr->line = line;
  chptr = nptr+sizeof(void*)+size;
  *chptr++ = END_PATTERN_1;
  *chptr++ = END_PATTERN_2;
  *chptr++ = END_PATTERN_3;
  *chptr++ = END_PATTERN_4;
#endif
#ifdef PAPI_DEBUG_MEMORY_TRACE
    fprintf(stdout, "0x%x: Re-allocated: %d bytes from File: %s  Line: %d\n", mem_ptr->ptr, size, file, line);
#endif
  return(mem_ptr->ptr);
}

void *_papi_calloc(char *file, int line, int nmemb, int size){
  void *ptr = _papi_malloc(file, line, size*nmemb);
#ifdef PAPI_DEBUG_MEMORY
  char *chptr;
#endif

  if ( !ptr ) return(NULL);
  memset(ptr, 0, size*nmemb);
#ifdef PAPI_DEBUG_MEMORY
  chptr = ptr+size;
  *chptr++ = END_PATTERN_1;
  *chptr++ = END_PATTERN_2;
  *chptr++ = END_PATTERN_3;
  *chptr++ = END_PATTERN_4;
  check_memory_for_overflow();
#endif
  return(ptr);
}

void *_papi_malloc(char *file, int line, int size){
  void *ptr;
  void **tmp;
  pmem_t *mem_ptr;
  int nsize = size+sizeof(void *);
#ifdef PAPI_DEBUG_MEMORY
  char *chptr;

  nsize += END_PAD;
#endif

  ptr = (void *) malloc(nsize);

  if ( !ptr ) return(NULL);
  else{
    if ( (mem_ptr = init_mem_ptr(ptr+sizeof(void *),size,file,line))==NULL) {
      free(ptr);
      return(NULL);
    }    
    tmp = ptr;
    *tmp = mem_ptr;
    ptr = mem_ptr->ptr;
    mem_ptr->ptr = ptr;
    _papi_hwi_lock(MEMORY_LOCK);
    insert_mem_ptr(mem_ptr);
    _papi_hwi_unlock(MEMORY_LOCK);

#ifdef PAPI_DEBUG_MEMORY
  chptr = ptr+size;
  *chptr++ = END_PATTERN_1;
  *chptr++ = END_PATTERN_2;
  *chptr++ = END_PATTERN_3;
  *chptr++ = END_PATTERN_4;
  check_memory_for_overflow();
#endif

#ifdef PAPI_DEBUG_MEMORY_TRACE
    fprintf(stdout, "0x%x: Allocated: %d bytes from File: %s  Line: %d\n", mem_ptr->ptr, size, file, line);
#endif
    return(ptr);
  }
  return(NULL);
}

/* Only frees the memory if PAPI malloced it */
void _papi_valid_free(char *file, int line, void *ptr){
  pmem_t *tmp;

  _papi_hwi_lock(MEMORY_LOCK);
  for(tmp = mem_head; tmp; tmp = tmp->next ){
    if ( ptr == tmp->ptr ){
      _papi_free(file, line, ptr);
      break;
    }
  }
  _papi_hwi_unlock(MEMORY_LOCK);
  return;
}

/* Frees up the ptr */
void _papi_free(char *file, int line, void *ptr){
  pmem_t *mem_ptr = get_mem_ptr(ptr);

  if ( !mem_ptr ) return;

#ifdef PAPI_DEBUG_MEMORY_TRACE
    fprintf(stdout, "0x%x: Freeing:   %d bytes from File: %s  Line: %d\n", mem_ptr->ptr, mem_ptr->size, file, line);
#endif
#ifdef PAPI_DEBUG_MEMORY
  check_memory_for_overflow();
#endif
  _papi_hwi_lock(MEMORY_LOCK);
  remove_mem_ptr(mem_ptr);
  _papi_hwi_unlock(MEMORY_LOCK);
}

/* Allocate and initialize a memory pointer */
pmem_t * init_mem_ptr(void *ptr, int size, char *file, int line){
 pmem_t *mem_ptr=NULL;
 if ((mem_ptr = (pmem_t *) malloc(sizeof(pmem_t)))==NULL)
   return(NULL);

 mem_ptr->ptr = ptr;
 mem_ptr->size = size;
 mem_ptr->next = NULL;
 mem_ptr->prev = NULL;
#ifdef PAPI_DEBUG_MEMORY
 strncpy(mem_ptr->file, file, DEBUG_FILE_LEN);
 mem_ptr->file[DEBUG_FILE_LEN-1] = '\0';
 mem_ptr->line = line;
#endif
 return(mem_ptr);
}

/* Print information about the memory including file and location it came from */
void _papi_mem_print_info(void *ptr) {
  pmem_t *mem_ptr = get_mem_ptr(ptr);

  if ( !mem_ptr ) return;
 
#ifdef PAPI_DEBUG_MEMORY
  printf("%p: Allocated %d bytes from file: %s  line: %d\n", ptr, mem_ptr->size, mem_ptr->file, mem_ptr->line);
#else
  printf("%p: Allocated %d bytes.", ptr, mem_ptr->size);
#endif
 return;
}

/* Print out all memory information */
void _papi_mem_print_stats(){
  pmem_t *tmp = NULL;

  _papi_hwi_lock(MEMORY_LOCK);
  for(tmp=mem_head;tmp;tmp = tmp->next){
#ifdef PAPI_DEBUG_MEMORY
  printf("%p: Allocated %d bytes from file: %s  line: %d\n", tmp->ptr, tmp->size, tmp->file, tmp->line);
#else
  printf("%p: Allocated %d bytes.\n", tmp->ptr, tmp->size);
#endif
  }
  _papi_hwi_unlock(MEMORY_LOCK);
}

/* Return the amount of memory overhead of the PAPI library and the memory system
 * PAPI_MEM_LIB_OVERHEAD is the library overhead
 * PAPI_MEM_OVERHEAD is the memory overhead
 * They both can be | together
 * This only includes "malloc'd memory"
 */
int papi_mem_overhead(int type){
  pmem_t *ptr=NULL;
  int size = 0;

  _papi_hwi_lock(MEMORY_LOCK);
  for(ptr=mem_head;ptr;ptr = ptr->next){
    if ( type&PAPI_MEM_LIB_OVERHEAD )
       size+=ptr->size;
    if ( type & PAPI_MEM_OVERHEAD ){
       size+=sizeof(pmem_t);
       size+=sizeof(void *);
#ifdef PAPI_DEBUG_MEMORY
       size+=END_PAD;
#endif
    }
  }
  _papi_hwi_unlock(MEMORY_LOCK);
  return(size);
}

/* Clean all memory up and print out memory leak information to stderr */
void _papi_cleanup_all_memory()
{
   pmem_t *ptr=NULL, *tmp=NULL;
#ifdef PAPI_DEBUG_MEMORY
   int cnt = 0;
#endif

   check_memory_for_overflow();
   _papi_hwi_lock(MEMORY_LOCK);
   for(ptr=mem_head;ptr;ptr=tmp){
     tmp = ptr->next;
#ifdef PAPI_DEBUG_MEMORY
     fprintf(stderr, "MEMORY LEAK: %p of %d bytes, from File: %s Line: %d\n", ptr->ptr, ptr->size, ptr->file, ptr->line);
     cnt += ptr->size;
#endif
     
     remove_mem_ptr(ptr);
   }
   _papi_hwi_unlock(MEMORY_LOCK);
#ifdef PAPI_DEBUG_MEMORY
   if ( cnt )
     fprintf(stderr, "TOTAL MEMORY LEAK: %d bytes.\n", cnt);
#endif
}

/* Insert the memory information 
 * Do not lock these routines, but lock in routines using these
 */
void insert_mem_ptr(pmem_t *ptr){
  if ( !ptr ) return;
 
  if ( !mem_head ) 
     mem_head = ptr;
  else {
     mem_head->prev = ptr;
     ptr->next = mem_head;
     mem_head = ptr;
  }
  return; 
}

/* Remove the memory information pointer and free the memory 
 * Do not using locking in this routine, instead lock around 
 * the sections of code that use this call.
 */
void remove_mem_ptr(pmem_t *ptr){
  if ( !ptr ) return;

  if ( ptr->prev )
    ptr->prev->next = ptr->next;
  if ( ptr->next )
    ptr->next->prev = ptr->prev;
  if ( ptr == mem_head )
      mem_head = ptr->next;
  free(ptr);
}

/* Check for memory buffer overflows */
int check_buffer_overflow(pmem_t *tmp){
#ifdef PAPI_DEBUG_OVERFLOW
  char *ptr;
  void *tptr;
  int fnd = 0;

  if ( !tmp ) return(0);

  tptr = tmp->ptr;
  tptr += tmp->size;

  /* Move to the buffer overflow padding */
  ptr = tmp->ptr+tmp->size;
  if ( *ptr++ != END_PATTERN_1 ) fnd=1;
  else if ( *ptr++ != END_PATTERN_2 ) fnd = 2;
  else if ( *ptr++ != END_PATTERN_3 ) fnd = 3;
  else if ( *ptr++ != END_PATTERN_4 ) fnd = 4;

  if ( fnd ) {
    fprintf(stderr, "Buffer Overflow[%d] for %p allocated from %s at line %d\n", fnd, tmp->ptr, tmp->file, tmp->line);
    return(1);
  }
#endif
  return(0);  
}

/* Loop through memory structures and look for buffer overflows 
 * returns the number of overflows detected
 */

int check_memory_for_overflow()
{
   int fnd = 0;
#ifdef PAPI_DEBUG_MEMORY
   pmem_t *tmp;

   _papi_hwi_lock(MEMORY_LOCK);
   for(tmp = mem_head; tmp; tmp = tmp->next){
     if ( check_buffer_overflow(tmp) ) fnd++;
   }
   if ( fnd )
     fprintf(stderr, "%d Total Buffer overflows detected!\n", fnd);
   _papi_hwi_unlock(MEMORY_LOCK);
#endif
   return(fnd);
}

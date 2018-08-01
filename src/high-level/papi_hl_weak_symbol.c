/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
* @file     papi_hl_weak_symbol.c
* @author   Frank Winkler
*           frank.winkler@icl.utk.edu
* @brief This file contains weak symbols for the high-level interface.
*/

#include <pthread.h>

int __attribute__((weak)) pthread_mutex_trylock(pthread_mutex_t *mutex)
{
   (void)(mutex);
   return 0;
}
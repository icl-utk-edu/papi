/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    threads.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
*/  

/* This file contains thread allocation and bookkeeping functions */

#ifdef _WIN32
  /* Define SUBSTRATE to map to linux-perfctr.h
   * since we haven't figured out how to assign a value 
   * to a label at make inside the Windows IDE */
  #define SUBSTRATE "linux-perfctr.h"
#endif

#include "papi.h"
#include SUBSTRATE
#include "papi_internal.h"
#include "papi_protos.h"

extern unsigned long int (*_papi_hwi_thread_id_fn)(void);

#ifdef ANY_THREAD_GETS_SIGNAL
extern int (*_papi_hwi_thread_kill_fn)(int, int);
#endif

static ThreadInfoList_t *head = NULL;

void _papi_hwi_shutdown_the_thread_list(void)
{
  ThreadInfoList_t *tmp;

  _papi_hwd_lock(PAPI_INTERNAL_LOCK);
  while (head)
    {
      tmp = head;
      head = head->next;
#ifdef THREAD_DEBUG
      fprintf(stderr,"%lld:%s:0x%x:Shutting down master thread %x at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),tmp->master->tid,tmp);
#endif
      _papi_hwd_shutdown(&tmp->master->context);
    }
  _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
}

void _papi_hwi_cleanup_thread_list(void)
{
  ThreadInfoList_t *tmp;

  _papi_hwd_lock(PAPI_INTERNAL_LOCK);
  while (head)
    {
      tmp = head;
      head = head->next;
      _papi_hwd_shutdown(&tmp->master->context);
#ifdef THREAD_DEBUG
      fprintf(stderr,"%lld:%s:0x%x:Freeing master thread %ld at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),tmp->master->tid,tmp);
#endif
      free(tmp);
    }
  _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
}

int _papi_hwi_insert_in_thread_list(ThreadInfo_t *ptr)
{
  ThreadInfoList_t *entry = (ThreadInfoList_t *)malloc(sizeof(ThreadInfoList_t));
  if (entry == NULL)
    return(PAPI_ENOMEM);
#ifdef THREAD_DEBUG
  fprintf(stderr,"%lld:%s:0x%x:(%p): New entry is at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),ptr,entry);
#endif
  entry->master = ptr;

  _papi_hwd_lock(PAPI_INTERNAL_LOCK);
  entry->next = head;
  head = entry;
#ifdef THREAD_DEBUG
  fprintf(stderr,"%lld:%s:0x%x:(%p): Old head is at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),ptr,entry->next);
  fprintf(stderr,"%lld:%s:0x%x:(%p): New head is at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),ptr,head);
#endif
  _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
  
  return(PAPI_OK);
}

ThreadInfo_t *_papi_hwi_lookup_in_thread_list(void)
{
  if (_papi_hwi_thread_id_fn == NULL)
    return(default_master_thread);
  else
    {
      unsigned long int id_to_find = (*_papi_hwi_thread_id_fn)();
      ThreadInfoList_t *tmp;

      _papi_hwd_lock(PAPI_INTERNAL_LOCK);
      tmp = head;
      while (tmp != NULL)
	{
#ifdef OVERFLOW_DEBUG
      fprintf(stderr,"%lld:%s:0x%x:Examining master at %p,tid 0x%x.\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),tmp->master,tmp->master->tid);
#endif
	  if (tmp->master->tid == id_to_find)
	    {
	      _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
	      return(tmp->master);
	    }
	  tmp = tmp->next;
	}
#ifdef OVERFLOW_DEBUG
      fprintf(stderr,"%lld:%s:0x%x:I'm not in the list at %p.\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),head);
#endif
      _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
      return(NULL);
    }
}



#ifdef ANY_THREAD_GETS_SIGNAL
void _papi_hwi_broadcast_overflow_signal(unsigned int mytid)
{
  int retval, didsomething = 0;
  ThreadInfoList_t *foo = NULL;
  _papi_hwd_lock(PAPI_INTERNAL_LOCK);
  for(foo = head ; foo != NULL; foo = foo->next ) 
    {
      if ((foo->master->event_set_overflowing) && (foo->master->tid != mytid))
	{
#ifdef OVERFLOW_DEBUG_TIMER
	  fprintf(stderr,"%lld:%s:0x%x:I'm forwarding signal to thread %x\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),foo->master->tid);
#endif
	  retval = (*_papi_hwi_thread_kill_fn)(foo->master->tid, PAPI_SIGNAL);
	  assert(retval == 0);
	  didsomething++;
	}
      else
	{
#ifdef OVERFLOW_DEBUG_TIMER
	  fprintf(stderr,"%lld:%s:0x%x:I'm NOT forwarding signal to thread %x\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),foo->master->tid);
#endif
	}
    }
  _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
  assert(didsomething);
}
#endif

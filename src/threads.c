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

/* The threads are now stored in a circular link-list, thus there is
 * no more locking on the lookup or insert function, but, one must
 * make sure if a remove function is put in that we again think about
 * locking, the shutdown/cleanup still lock
 */
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

/*
 * Calls _papi_hwd_shutdown on every element of the thread list
 * but does not free up the memory.
 */
void _papi_hwi_shutdown_the_thread_list(void)
{
  ThreadInfoList_t *tmp;
  ThreadInfoList_t *nxt;

  _papi_hwd_lock(PAPI_INTERNAL_LOCK);
  if ( head == NULL ){
        _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
	return;
  }

  tmp = head->next;
  while (tmp != head)
  {
      nxt = tmp->next;
#ifdef THREAD_DEBUG
      fprintf(stderr,"%lld:%s:0x%x:Shutting down master thread %x at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),tmp->master->tid,tmp);
#endif
      _papi_hwd_shutdown(&tmp->master->context);
      tmp = nxt;
  }
  if ( tmp ){
#ifdef THREAD_DEBUG
      fprintf(stderr,"%lld:%s:0x%x:Shutting down master thread %x at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),tmp->master->tid,tmp);
#endif
      _papi_hwd_shutdown(&tmp->master->context);
  }
  _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
}

/*
 * Free's up any memory associated with the list
 */
void _papi_hwi_cleanup_thread_list(void)
{
  ThreadInfoList_t *tmp;
  ThreadInfoList_t *nxt;
  ThreadInfoList_t *prev;

  _papi_hwd_lock(PAPI_INTERNAL_LOCK);
  if ( head == NULL ){
        _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
	return;
  }

  prev = head;
  tmp = head->next;
  while (tmp != head)
  {
      nxt = tmp->next;
#ifdef THREAD_DEBUG
      fprintf(stderr,"%lld:%s:0x%x:Freeing master thread %ld at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),tmp->master->tid,tmp);
      fprintf(stderr,"%p->%p: %p->%p: %p->%p", prev, prev->next,tmp, tmp->next,nxt,nxt->next);
#endif
      free(tmp);
      prev->next = nxt;
      tmp = nxt;
  }
#ifdef THREAD_DEBUG
  fprintf(stderr,"%lld:%s:0x%x:Freeing master thread %ld at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),tmp->master->tid,tmp);
#endif
  free(tmp);
  head = NULL;
  _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
}

int _papi_hwi_insert_in_thread_list(ThreadInfo_t *ptr)
{
  ThreadInfoList_t *entry = (ThreadInfoList_t *)malloc(sizeof(ThreadInfoList_t));
  int i;

  if (entry == NULL)
    return(PAPI_ENOMEM);
#ifdef THREAD_DEBUG
  fprintf(stderr,"%lld:%s:0x%x:(%p): New entry is at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),ptr,entry);
#endif
  entry->master = ptr;

  /* Thread specific data */
  for ( i=0; i < PAPI_MAX_THREAD_STORAGE; i++ )
	ptr->thread_storage[0] = NULL;

  /*_papi_hwd_lock(PAPI_INTERNAL_LOCK);*/
  if ( head == NULL ){
     head = entry;
     head->next = head;
  }
  else{
    /* Since we are no longer locking the entry has to be setup BEFORE
     * inserting it into the circular list -KSL
     */
    entry->next = head->next;
    head->next = entry;
    head = entry;
  }
#ifdef THREAD_DEBUG
  fprintf(stderr,"%lld:%s:0x%x:(%p): Old head is at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),ptr,entry->next);
  fprintf(stderr,"%lld:%s:0x%x:(%p): New head is at %p\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),ptr,head);
#endif
  /*_papi_hwd_unlock(PAPI_INTERNAL_LOCK);*/
  
  return(PAPI_OK);
}

inline ThreadInfo_t *_papi_hwi_lookup_in_thread_list(void)
{
  if (_papi_hwi_thread_id_fn == NULL)
    return(default_master_thread);
  else
    {
      unsigned long int id_to_find = (*_papi_hwi_thread_id_fn)();
      ThreadInfoList_t *tmp;

      /*_papi_hwd_lock(PAPI_INTERNAL_LOCK);*/
      tmp = head;
      while (tmp != NULL)
	{
#ifdef OVERFLOW_DEBUG
      fprintf(stderr,"%lld:%s:0x%x:Examining master at %p,tid 0x%x.\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),tmp->master,tmp->master->tid);
#endif
	  if (tmp->master->tid == id_to_find)
	    {
	      /*_papi_hwd_unlock(PAPI_INTERNAL_LOCK);*/
	      head = tmp;
	      return(tmp->master);
	    }
	  tmp = tmp->next;
          if ( tmp == head )
	     break;
	}
#ifdef OVERFLOW_DEBUG
      fprintf(stderr,"%lld:%s:0x%x:I'm not in the list at %p.\n",_papi_hwd_get_real_usec(),__FUNCTION__,(*_papi_hwi_thread_id_fn)(),head);
#endif
      /*_papi_hwd_unlock(PAPI_INTERNAL_LOCK);*/
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
       if( foo->next == head )
	 break;
    }
  _papi_hwd_unlock(PAPI_INTERNAL_LOCK);
  assert(didsomething);
}
#endif

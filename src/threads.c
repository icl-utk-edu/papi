/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    threads.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    <your name here>
*          <your email here>
*/  

/* This file contains thread allocation and bookkeeping functions */

#include "papi.h"

#ifndef _WIN32
  #include SUBSTRATE
#else
  #include "win32.h"
#endif

#include "papi_internal.h"

#include "papi_protos.h"

static ThreadInfoList_t *head = NULL;

void _papi_hwi_cleanup_thread_list(void)
{
  ThreadInfoList_t *tmp;

  _papi_hwd_lock();
  while (head)
    {
      tmp = head;
      head = head->next;
      _papi_hwd_shutdown(&tmp->master->context);
      DBG((stderr,"Freeing master thread %d at %p\n",tmp->master->tid,tmp));
      free(tmp);
    }
  _papi_hwd_unlock();
}

int _papi_hwi_insert_in_thread_list(ThreadInfo_t *ptr)
{
  ThreadInfoList_t *entry = (ThreadInfoList_t *)malloc(sizeof(ThreadInfoList_t));
  if (entry == NULL)
    return(PAPI_ENOMEM);
  DBG((stderr,"(%p): New entry is at %p\n",ptr,entry));
  entry->master = ptr;

  _papi_hwd_lock();
  entry->next = head;
  DBG((stderr,"(%p): Old head is at %p\n",ptr,entry->next));
  head = entry;
  DBG((stderr,"(%p): New head is at %p\n",ptr,head));
  _papi_hwd_unlock();
  
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

      _papi_hwd_lock();
      tmp = head;
      while (tmp != NULL)
	{
	  if (tmp->master->tid == id_to_find)
	    {
	      _papi_hwd_unlock();
	      return(tmp->master);
	    }
	  tmp = tmp->next;
	}
      DBG((stderr,"New thread %lu found, but not initialized.\n",id_to_find));
      _papi_hwd_unlock();
      return(NULL);
    }
}


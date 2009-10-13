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

#include "papi.h"
#include "papi_internal.h"
#include "papi_memory.h"

/*****************/
/* BEGIN GLOBALS */
/*****************/

/* The following globals get initialized and cleared by:
   extern int _papi_hwi_init_global_threads(void);
   extern int _papi_hwi_shutdown_thread(ThreadInfo_t *thread); */

/* list of threads, gets initialized to master process with TID of getpid() */

volatile ThreadInfo_t *_papi_hwi_thread_head;

/* If we have TLS, this variable ALWAYS points to our thread descriptor. It's like magic! */

#if defined(HAVE_THREAD_LOCAL_STORAGE)
THREAD_LOCAL_STORAGE_KEYWORD ThreadInfo_t *_papi_hwi_my_thread;
#endif

/* Function that returns and unsigned long thread identifier */

unsigned long (*_papi_hwi_thread_id_fn)(void);

/* Function that sends a signal to other threads */

#ifdef ANY_THREAD_GETS_SIGNAL
int (*_papi_hwi_thread_kill_fn)(int, int);
#endif

/*****************/
/*  END  GLOBALS */
/*****************/

static int lookup_and_set_thread_symbols(void)
{
#if defined(ANY_THREAD_GETS_SIGNAL)
   int retval;
   char *error_ptc = NULL, *error_ptk = NULL;
   void *symbol_ptc = NULL, *symbol_ptk = NULL, *handle = NULL;

   handle = dlopen(NULL, RTLD_LAZY);
   if (handle == NULL)
     {
       PAPIERROR("Error from dlopen(NULL, RTLD_LAZY): %d %s",errno,dlerror());
       return(PAPI_ESYS);
     }

   symbol_ptc = dlsym(handle, "pthread_self");
   if (symbol_ptc == NULL)
     {
       error_ptc = dlerror();
       THRDBG("dlsym(%p,pthread_self) returned NULL: %s\n",(error_ptc ? error_ptc : "No error, NULL symbol!"));
     }
   
   symbol_ptk = dlsym(handle, "pthread_kill");
   if (symbol_ptk == NULL)
     {
       error_ptk = dlerror();
       THRDBG("dlsym(%p,pthread_kill) returned NULL: %s\n",(error_ptk ? error_ptk : "No error, NULL symbol!"));
     }
	  
   dlclose(handle);

   if (!((_papi_hwi_thread_kill_fn && _papi_hwi_thread_id_fn) ||
	 (!_papi_hwi_thread_kill_fn && !_papi_hwi_thread_id_fn)))
     return(PAPI_EMISC);

   _papi_hwi_thread_kill_fn = (int (*)(int,int))symbol_ptk;
   _papi_hwi_thread_id_fn = (unsigned long (*)(void))symbol_ptc;
#endif
   return(PAPI_OK);
}

static ThreadInfo_t *allocate_thread(void)
{
   ThreadInfo_t *thread;
   int i;

   /* The Thread EventSet is special. It is not in the EventSet list, but is pointed
      to by each EventSet of that particular thread. */

   thread = (ThreadInfo_t *) papi_malloc(sizeof(ThreadInfo_t));
   if (thread == NULL)
      return (NULL);
   memset(thread, 0x00, sizeof(ThreadInfo_t));

   thread->context = (hwd_context_t **) papi_malloc(sizeof(hwd_context_t *)*papi_num_components);
   if ( !thread->context ){
     papi_free(thread);
     return(NULL);
   }

   thread->running_eventset = (EventSetInfo_t **) papi_malloc(sizeof(EventSetInfo_t *)*papi_num_components);
   if ( !thread->running_eventset ){
     papi_free(thread->context);
     papi_free(thread);
     return(NULL);
   }

   for(i=0;i<papi_num_components;i++ ){
     thread->context[i] = (void *) papi_malloc(_papi_hwd[i]->size.context);
     thread->running_eventset[i] = NULL;
     if ( thread->context[i] == NULL ){
       for(i--;i>=0;i--)
         papi_free(thread->context[i]);
       papi_free(thread->context);
       papi_free(thread);
       return(NULL);
     }
     memset(thread->context[i], 0x00, _papi_hwd[i]->size.context);
    }


   if (_papi_hwi_thread_id_fn)
     thread->tid = (*_papi_hwi_thread_id_fn)();
   else
     thread->tid = getpid();

   THRDBG("Allocated thread 0x%lx at %p\n",thread->tid,thread);

   return (thread);
}

static void free_thread(ThreadInfo_t ** thread)
{
   int i;
   THRDBG("Freeing thread 0x%lx at %p\n",(*thread)->tid,*thread);

   for(i=0;i<papi_num_components;i++) {
     if ( (*thread)->context[i] )
       papi_free((*thread)->context[i]);
   }

   if ( (*thread)->context )
      papi_free((*thread)->context);
 
   if ( (*thread)->running_eventset )
      papi_free((*thread)->running_eventset);

   memset(*thread, 0x00, sizeof(ThreadInfo_t));
   papi_free(*thread);
   *thread = NULL;
}

static void insert_thread (ThreadInfo_t * entry)
{
  _papi_hwi_lock (THREADS_LOCK);	
  
  if (_papi_hwi_thread_head == NULL) /* 0 elements */
    {
      THRDBG ("_papi_hwi_thread_head is NULL\n");
      entry->next = entry;
    }
  else if (_papi_hwi_thread_head->next == _papi_hwi_thread_head) /* 1 elements */
    {
      THRDBG ("_papi_hwi_thread_head was thread 0x%lx at %p\n",_papi_hwi_thread_head->tid,_papi_hwi_thread_head);
      _papi_hwi_thread_head->next = entry;
      entry->next = (ThreadInfo_t *)_papi_hwi_thread_head;
    }
  else /* 2+ elements */
    {
      THRDBG ("_papi_hwi_thread_head was thread 0x%lx at %p\n",_papi_hwi_thread_head->tid,_papi_hwi_thread_head);
      entry->next = _papi_hwi_thread_head->next;
      _papi_hwi_thread_head->next = entry;
    }

  _papi_hwi_thread_head = entry;

  THRDBG("_papi_hwi_thread_head now thread 0x%lx at %p\n",_papi_hwi_thread_head->tid,_papi_hwi_thread_head);

  _papi_hwi_unlock (THREADS_LOCK);	

#if defined(HAVE_THREAD_LOCAL_STORAGE)
  _papi_hwi_my_thread = entry;
  THRDBG("TLS for thread 0x%lx is now %p\n",entry->tid,_papi_hwi_my_thread);
#endif
}

static int remove_thread (ThreadInfo_t * entry)
{
  ThreadInfo_t *tmp = NULL, *prev = NULL;

  _papi_hwi_lock (THREADS_LOCK);	

  THRDBG ("_papi_hwi_thread_head was thread 0x%lx at %p\n",_papi_hwi_thread_head->tid,_papi_hwi_thread_head);

  /* Find the preceding element and the matched element,
     short circuit if we've seen the head twice */

  for (tmp = (ThreadInfo_t *)_papi_hwi_thread_head; (entry != tmp) || (prev == NULL); tmp = tmp->next)
    {
      prev = tmp;
    }

  if (tmp != entry)
    {
      THRDBG("Thread 0x%lx at %p was not found in the thread list!\n",entry->tid,entry);
      return(PAPI_EBUG);
    }

  /* Only 1 element in list */

  if (prev == tmp)
    {
      _papi_hwi_thread_head = NULL;
      tmp->next = NULL;
      THRDBG("_papi_hwi_thread_head now NULL\n");
    }
  else
    {
      prev->next = tmp->next;
      /* If we're removing the head, better advance it! */
      if (_papi_hwi_thread_head == tmp)
	{
	  _papi_hwi_thread_head = tmp->next;
	  THRDBG("_papi_hwi_thread_head now thread 0x%lx at %p\n",_papi_hwi_thread_head->tid,_papi_hwi_thread_head);
	}
      THRDBG("Removed thread %p from list\n",tmp);
    }

  _papi_hwi_unlock (THREADS_LOCK);	

#if defined(HAVE_THREAD_LOCAL_STORAGE)
  _papi_hwi_my_thread = NULL;
  THRDBG("TLS for thread 0x%lx is now %p\n",entry->tid,_papi_hwi_my_thread);
#endif

  return (PAPI_OK);
}

int _papi_hwi_initialize_thread(ThreadInfo_t ** dest)
{
   int retval;
   ThreadInfo_t *thread;
   int i;

   if ((thread = allocate_thread()) == NULL)
     {
       *dest = NULL;
       return(PAPI_ENOMEM);
     }
   
   /* Call the substrate to fill in anything special. */

   for ( i=0;i<papi_num_components;i++){
     retval=_papi_hwd[i]->init(thread->context[i]);
     if (retval) 
       {
         free_thread(&thread);
         *dest = NULL;
         return (retval);
       }
    }

   insert_thread(thread);

   *dest = thread;
   return (PAPI_OK);
}

#if defined(ANY_THREAD_GETS_SIGNAL)

/* This is ONLY defined for systems that enable ANY_THREAD_GETS_SIGNAL
   since we must forward signals sent to non-PAPI threads.

   This is NOT compatible with thread local storage, since to broadcast
   the signal, we need a list of threads. */

int _papi_hwi_broadcast_signal(unsigned int mytid)
{
  int i, retval, didsomething = 0;
  volatile ThreadInfo_t *foo = NULL;

  _papi_hwi_lock(THREADS_LOCK);

  for (foo = _papi_hwi_thread_head; foo != NULL; foo = foo->next)
    {
	/* xxxx Should this be hardcoded to index 0 or walk the list or what? */
   for ( i=0;i<papi_num_components;i++){
      if ((foo->tid != mytid) && (foo->running_eventset[i]) && 
	  (foo->running_eventset[i]->state & (PAPI_OVERFLOWING|PAPI_MULTIPLEXING)))
	{
	/* xxxx mpx_info inside _papi_mdi_t _papi_hwi_system_info is commented out.
		See papi_internal.h for details. The multiplex_timer_sig value is now part of that structure */
/*
	  THRDBG("Thread 0x%lx sending signal %d to thread 0x%lx\n",mytid,foo->tid,
		  (foo->running_eventset[i]->state & PAPI_OVERFLOWING ? _papi_hwd[i]->cmp_info.hardware_intr_sig : _papi_hwd[i]->cmp_info.multiplex_timer_sig));
	  retval = (*_papi_hwi_thread_kill_fn)(foo->tid, 
		  (foo->running_eventset[i]->state & PAPI_OVERFLOWING ? _papi_hwd[i]->cmp_info.hardware_intr_sig : _papi_hwd[i]->cmp_info.multiplex_timer_sig));
	  if (retval != 0)
	    return(PAPI_EMISC);
*/
	  }
      if (foo->next == _papi_hwi_thread_head)
	break;
   }
  }
  _papi_hwi_unlock (THREADS_LOCK);

  return(PAPI_OK);
}
#endif

/* This is undefined for systems that enable ANY_THREAD_GETS_SIGNAL
   since we always must enable threads for safety. */

int _papi_hwi_set_thread_id_fn(unsigned long (*id_fn) (void))
{
#if !defined(ANY_THREAD_GETS_SIGNAL)
  /* Check for multiple threads still in the list, if so, we can't change it */

  if (_papi_hwi_thread_head->next != _papi_hwi_thread_head)
    return(PAPI_EINVAL);
  
  /* We can't change the thread id function from one to another, 
     only NULL to non-NULL and vice versa. */

  if ((id_fn != NULL) && (_papi_hwi_thread_id_fn != NULL))
    return(PAPI_EINVAL);
  
  _papi_hwi_thread_id_fn = id_fn;

  THRDBG("Set new thread id function to %p\n",id_fn);

  if (id_fn)
    _papi_hwi_thread_head->tid = (*_papi_hwi_thread_id_fn)();
  else
    _papi_hwi_thread_head->tid = getpid();

  THRDBG("New master tid is 0x%lx\n",_papi_hwi_thread_head->tid);
#else
  THRDBG("Skipping set of thread id function\n");
#endif

  return(PAPI_OK);
}

int _papi_hwi_shutdown_thread(ThreadInfo_t *thread)
{
   int retval = PAPI_OK;
   unsigned long tid;
   int i, failure=0;

   if (_papi_hwi_thread_id_fn)
     tid = (*_papi_hwi_thread_id_fn)();
   else
     tid = getpid();

   if (thread->tid == tid)
     {
       remove_thread(thread);
       THRDBG ("Shutting down thread 0x%lx at %p\n",thread->tid,thread);
       for(i=0;i<papi_num_components;i++){
          retval = _papi_hwd[i]->shutdown(thread->context[i]);
          if ( retval != PAPI_OK) failure=retval;
       }
       free_thread(&thread);
       return(failure);
     }

   THRDBG("Skipping shutdown thread 0x%lx at %p, thread 0x%lx not owner!\n",thread->tid,thread,tid);
   return (PAPI_EBUG);
}

/* THESE MUST BE CALLED WITH A GLOBAL LOCK */

int _papi_hwi_shutdown_global_threads(void)
{
  int err;
  ThreadInfo_t *tmp = _papi_hwi_lookup_thread();

  if (tmp == NULL)
    {
      THRDBG("Did not find my thread for shutdown!\n");      
      err = PAPI_EBUG;
    }
  else
    {
      err = _papi_hwi_shutdown_thread(tmp);
    }

#ifdef DEBUG
  if (ISLEVEL(DEBUG_THREADS))
    {
      if (_papi_hwi_thread_head)
	{
	  THRDBG("Thread head %p still exists!\n",_papi_hwi_thread_head);
	}
    }
#endif

#if defined(HAVE_THREAD_LOCAL_STORAGE)
  _papi_hwi_my_thread = NULL;
#endif
  _papi_hwi_thread_head = NULL;
  _papi_hwi_thread_id_fn = NULL;
#if defined(ANY_THREAD_GETS_SIGNAL)
  _papi_hwi_thread_kill_fn = NULL;
#endif

  return(err);
}

int _papi_hwi_init_global_threads(void)
{
  int retval;
  ThreadInfo_t *tmp;

  _papi_hwi_lock(GLOBAL_LOCK);
  
#if defined(HAVE_THREAD_LOCAL_STORAGE)
  _papi_hwi_my_thread = NULL;
#endif
  _papi_hwi_thread_head = NULL;
  _papi_hwi_thread_id_fn = NULL;
#if defined(ANY_THREAD_GETS_SIGNAL)
  _papi_hwi_thread_kill_fn = NULL;
#endif

  retval = _papi_hwi_initialize_thread(&tmp);
  if (retval == PAPI_OK)
    retval = lookup_and_set_thread_symbols();

  _papi_hwi_unlock(GLOBAL_LOCK);

  return(retval);
}

int _papi_hwi_gather_all_thrspec_data(int tag, PAPI_all_thr_spec_t *where)
{
  int didsomething = 0;
  ThreadInfo_t *foo = NULL;

  _papi_hwi_lock(THREADS_LOCK);

  for (foo = (ThreadInfo_t *)_papi_hwi_thread_head; foo != NULL; foo = foo->next)
    {
      /* If we want thread ID's */
      if (where->id)
	memcpy(&where->id[didsomething],&foo->tid,sizeof(where->id[didsomething]));

      /* If we want data pointers */
      if (where->data)
	where->data[didsomething] = foo->thread_storage[tag];

      didsomething++;

      if ((where->id) || (where->data))
	{
	  if (didsomething >= where->num)
	    break;
	}

      if (foo->next == _papi_hwi_thread_head)
	break;
    }

  where->num = didsomething;
  _papi_hwi_unlock (THREADS_LOCK);

  return(PAPI_OK);

}

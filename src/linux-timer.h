long long _linux_get_real_usec( void );
long long _linux_get_real_cycles( void );
long long _linux_get_virt_usec( hwd_context_t *zero );
long long _linux_get_virt_cycles( hwd_context_t *zero );
int mmtimer_setup(void);
int init_proc_thread_timer( hwd_context_t *thr_ctx );


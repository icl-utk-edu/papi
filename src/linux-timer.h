long long _linux_get_real_cycles( void );

long long _linux_get_virt_usec_pttimers( hwd_context_t *zero );
long long _linux_get_virt_usec_gettime( hwd_context_t *zero );
long long _linux_get_virt_usec_timess( hwd_context_t *zero );
long long _linux_get_virt_usec_rusages( hwd_context_t *zero );

long long _linux_get_real_usec_gettime( void );
long long _linux_get_real_usec_gettimeofday( void );
long long _linux_get_real_usec_fallback( void );


int mmtimer_setup(void);
int init_proc_thread_timer( hwd_context_t *thr_ctx );


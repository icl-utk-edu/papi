#ifndef __SHM_H__
#define __SHM_H__

int shm_init( char *status );
int shm_shutdown( void );
int shm_alloc( int shm_elem_size, int *shm_elem_count, int *shm_handle );
int shm_free( int shm_handle );
int shm_put( int shm_handle, int local_proc_id, void *data, int size );
int shm_get( int shm_handle, int local_proc_id, void *data, int size );
int shm_get_local_proc_id( void );
int shm_get_global_proc_id( void );

#endif /* End of __SHM_H__ */

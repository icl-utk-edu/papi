#include <string.h>
#include <dlfcn.h>
#include "shm.h"
#include "sysdetect.h"

#ifdef HAVE_MPI
#include <mpi.h>

int (*MPI_InitializedPtr)( int *flag ) = NULL;
int (*MPI_Comm_sizePtr)( MPI_Comm comm, int *size ) = NULL;
int (*MPI_Comm_rankPtr)( MPI_Comm comm, int *rank ) = NULL;
int (*MPI_Win_allocate_sharedPtr)( MPI_Aint size, int disp_unit,
                                   MPI_Info info, MPI_Comm comm,
                                   void *baseptr, MPI_Win *win ) = NULL;
int (*MPI_Win_freePtr)( MPI_Win *win ) = NULL;
int (*MPI_Win_shared_queryPtr)( MPI_Win win, int rank, MPI_Aint *size,
                                int *disp_unit, void *baseptr ) = NULL;
int (*MPI_Win_lock_allPtr)( int assert, MPI_Win win ) = NULL;;
int (*MPI_Win_unlock_allPtr)( MPI_Win win ) = NULL;
int (*MPI_Win_syncPtr)( MPI_Win win ) = NULL;
int (*MPI_Comm_split_typePtr)( MPI_Comm comm, int split_type, int key,
                               MPI_Info info, MPI_Comm *new_comm ) = NULL;
int (*MPI_BarrierPtr)( MPI_Comm comm ) = NULL;

#define MPI_CALL(call, err_handle) do { \
    int _status = (call);               \
    if (_status == MPI_SUCCESS)         \
        break;                          \
    err_handle;                         \
} while(0)

typedef struct {
    MPI_Win win;
    void *seg_p;
} shm_seg_t;

static struct {
    int capacity;
    int seg_count;
    shm_seg_t *seg_arr;
} shm_area;

static MPI_Comm shm_comm;
static void *dlp_mpi = NULL;

static int mpi_is_enabled( void );
static int load_mpi_sym( char * );
static int unload_mpi_sym( void );
static int get_free_shm_handle( void );

int
shm_init( char *status )
{
    if (load_mpi_sym(status)) {
        SUBDBG("Warning: MPI library not found.\n");
        return -1;
    }

    int mpi_initialized;
    MPI_CALL((*MPI_InitializedPtr)(&mpi_initialized), );
    if (!mpi_initialized) {
        const char *message = "MPI not initialized";
        int count = snprintf(status, strlen(message) + 1, message);
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        SUBDBG("Error: MPI library is not initialized.\n");
        return -1;
    }

    return 0;
}

int
shm_shutdown( void )
{
    return unload_mpi_sym();
}

int
shm_alloc( int shm_elem_size, int *shm_elem_count, int *shm_handle )
{
    int rank;

    MPI_CALL((*MPI_Comm_rankPtr)(MPI_COMM_WORLD, &rank), return _status);

    MPI_CALL((*MPI_Comm_split_typePtr)(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                       rank, MPI_INFO_NULL, &shm_comm),
             return _status);

    *shm_handle = get_free_shm_handle();
    shm_area.seg_arr[*shm_handle].win = MPI_WIN_NULL;

    int shm_comm_size;
    MPI_CALL((*MPI_Comm_sizePtr)(shm_comm, &shm_comm_size), return _status);

    MPI_CALL((* MPI_Win_allocate_sharedPtr)(shm_comm_size * shm_elem_size,
                                            shm_elem_size, MPI_INFO_NULL, shm_comm,
                                            &shm_area.seg_arr[*shm_handle].seg_p,
                                            &shm_area.seg_arr[*shm_handle].win),
             return _status);

    *shm_elem_count = shm_comm_size;

    return 0;
}

int
shm_free( int shm_handle )
{
    /* sanity check */
    if (shm_handle >= shm_area.seg_count || shm_handle < 0)
        return -1;

    MPI_CALL((*MPI_Win_freePtr)(&shm_area.seg_arr[shm_handle].win), return _status);
    shm_area.seg_arr[shm_handle].seg_p = NULL;
    shm_area.seg_count -= 1;

    if (shm_area.seg_count == 0) {
        papi_free(shm_area.seg_arr);
        shm_area.seg_arr = NULL;
        shm_area.capacity = 0;
    }

    return 0;
}

int
shm_put( int shm_handle, int local_proc_id, void *data, int size )
{
    /* sanity check */
    if (shm_handle >= shm_area.seg_count || shm_handle < 0)
        return -1;

    MPI_Aint ret_size;
    int ret_disp_unit;
    void *target_ptr = NULL;
    MPI_CALL((*MPI_Win_shared_queryPtr)(shm_area.seg_arr[shm_handle].win,
                                        local_proc_id, &ret_size,
                                        &ret_disp_unit, (void *) &target_ptr),
             return _status);

    MPI_CALL((*MPI_Win_lock_allPtr)(MPI_MODE_NOCHECK,
                                    shm_area.seg_arr[shm_handle].win),
             return _status);

    /* Do store to shared memory */
    memcpy(target_ptr, data, size);

    /* Sync shared memory */
    MPI_CALL((*MPI_Win_syncPtr)(shm_area.seg_arr[shm_handle].win), return _status);
    MPI_CALL((*MPI_Win_unlock_allPtr)(shm_area.seg_arr[shm_handle].win),
             return _status);

    /* Wait for all processes to update shared memory */
    MPI_CALL((*MPI_BarrierPtr)(shm_comm), return _status);

    return 0;
}

int
shm_get( int shm_handle, int local_proc_id, void *data, int size )
{
    /* sanity check */
    if (shm_handle >= shm_area.seg_count || shm_handle < 0)
        return -1;

    MPI_Aint ret_size;
    int ret_disp_unit;
    void *target_ptr = NULL;
    MPI_CALL((*MPI_Win_shared_queryPtr)(shm_area.seg_arr[shm_handle].win,
                                        local_proc_id, &ret_size,
                                        &ret_disp_unit, (void *)&target_ptr),
             return _status);

    memcpy(data, target_ptr, size);

    return 0;
}

int
shm_get_local_proc_id( void )
{
    int shm_rank;
    MPI_CALL((*MPI_Comm_rankPtr)(shm_comm, &shm_rank), return -1);
    return shm_rank;
}

int
shm_get_global_proc_id( void )
{
    int world_rank;
    MPI_CALL((*MPI_Comm_rankPtr)(MPI_COMM_WORLD, &world_rank), return -1);
    return world_rank;
}

int
mpi_is_enabled( void )
{
    return (MPI_InitializedPtr         != NULL &&
            MPI_Comm_sizePtr           != NULL &&
            MPI_Comm_rankPtr           != NULL &&
            MPI_Win_allocate_sharedPtr != NULL &&
            MPI_Win_freePtr            != NULL &&
            MPI_Win_shared_queryPtr    != NULL &&
            MPI_Win_lock_allPtr        != NULL &&
            MPI_Win_unlock_allPtr      != NULL &&
            MPI_Win_syncPtr            != NULL &&
            MPI_Comm_split_typePtr     != NULL &&
            MPI_BarrierPtr             != NULL);
}

int
load_mpi_sym( char *status )
{
    dlp_mpi = dlopen("libmpi.so", RTLD_NOW | RTLD_GLOBAL);
    if (dlp_mpi == NULL) {
        int count = snprintf(status, PAPI_MAX_STR_LEN, "%s", dlerror());
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        status[PAPI_MAX_STR_LEN - 1] = 0;
        return -1;
    }

    MPI_InitializedPtr         = dlsym(dlp_mpi, "MPI_Initialized");
    MPI_Comm_sizePtr           = dlsym(dlp_mpi, "MPI_Comm_size");
    MPI_Comm_rankPtr           = dlsym(dlp_mpi, "MPI_Comm_rank");
    MPI_Win_allocate_sharedPtr = dlsym(dlp_mpi, "MPI_Win_allocate_shared");
    MPI_Win_freePtr            = dlsym(dlp_mpi, "MPI_Win_free");
    MPI_Win_shared_queryPtr    = dlsym(dlp_mpi, "MPI_Win_shared_query");
    MPI_Win_lock_allPtr        = dlsym(dlp_mpi, "MPI_Win_lock_all");
    MPI_Win_unlock_allPtr      = dlsym(dlp_mpi, "MPI_Win_unlock_all");
    MPI_Win_syncPtr            = dlsym(dlp_mpi, "MPI_Win_sync");
    MPI_Comm_split_typePtr     = dlsym(dlp_mpi, "MPI_Comm_split_type");
    MPI_BarrierPtr             = dlsym(dlp_mpi, "MPI_Barrier");

    if (!mpi_is_enabled()) {
        const char *message = "dlsym() of MPI symbols failed";
        int count = snprintf(status, strlen(message) + 1, message);
        if (count >= PAPI_MAX_STR_LEN) {
            SUBDBG("Status string truncated.");
        }
        return -1;
    }

    return 0;
}

int
unload_mpi_sym( void )
{
    if (dlp_mpi != NULL) {
        dlclose(dlp_mpi);
    }

    MPI_InitializedPtr         = NULL;
    MPI_Comm_sizePtr           = NULL;
    MPI_Comm_rankPtr           = NULL;
    MPI_Win_allocate_sharedPtr = NULL;
    MPI_Win_freePtr            = NULL;
    MPI_Win_shared_queryPtr    = NULL;
    MPI_Win_lock_allPtr        = NULL;
    MPI_Win_unlock_allPtr      = NULL;
    MPI_Win_syncPtr            = NULL;
    MPI_Comm_split_typePtr     = NULL;
    MPI_BarrierPtr             = NULL;

    return mpi_is_enabled();
}

int
get_free_shm_handle( void )
{
    int shm_handle;

    if (shm_area.capacity == shm_area.seg_count) {
        /* no more space; expand segment array by 1 element */
        shm_area.capacity += 1;
        shm_area.seg_arr = papi_realloc(shm_area.seg_arr,
                                        sizeof(shm_seg_t) * shm_area.capacity);
        shm_handle = shm_area.seg_count++;
    } else {
        /* find free element */
        shm_handle = -1;
        while (shm_area.seg_arr[++shm_handle].seg_p);
    }

    return shm_handle;
}
#else
static char *local_mem;

int
shm_init( char *status )
{
    const char *message = "MPI not configured";
    int count = snprintf(status, strlen(message) + 1, message);
    if (count >= PAPI_MAX_STR_LEN) {
        SUBDBG("Status string truncated.");
    }
    return 0;
}

int
shm_shutdown( void )
{
    return 0;
}

int
shm_alloc( int shm_elem_size, int *shm_elem_count, int *shm_handle )
{
    *shm_elem_count = 1;
    *shm_handle = 0;
    local_mem = papi_calloc(1, shm_elem_size);
    return 0;
}

int
shm_free( int shm_handle )
{
    free(&local_mem[shm_handle]);
    return 0;
}

int
shm_put( int shm_handle, int local_proc_id __attribute__((unused)),
         void *data, int size )
{
    memcpy(&local_mem[shm_handle], data, size);
    return 0;
}

int
shm_get( int shm_handle, int local_proc_id __attribute__((unused)),
         void *data, int size )
{
    memcpy(data, &local_mem[shm_handle], size);
    return 0;
}

int
shm_get_local_proc_id( void )
{
    return 0;
}

int
shm_get_global_proc_id( void )
{
    return 0;
}
#endif /* HAVE_MPI */

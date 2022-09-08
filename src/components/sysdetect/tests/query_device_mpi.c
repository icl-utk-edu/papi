#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "papi.h"
#include "papi_test.h"

int main(int argc, char *argv[])
{
    int quiet = 0;
    quiet = tests_quiet(argc, argv);

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed\n", retval);
    }

    if (!quiet && rank == 0) {
        printf("Testing systedect component with PAPI %d.%d.%d\n",
               PAPI_VERSION_MAJOR(PAPI_VERSION),
               PAPI_VERSION_MINOR(PAPI_VERSION),
               PAPI_VERSION_REVISION(PAPI_VERSION));
    }

    void *handle;
    int enum_modifier = PAPI_DEV_TYPE_ENUM__ALL;
    int i, id;
    const char *name = NULL;
    int dev_count;

    while (PAPI_enum_dev_type(enum_modifier, &handle) == PAPI_OK) {
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_PAPI_ID, &id);
        PAPI_get_dev_type_attr(handle, PAPI_DEV_TYPE_ATTR__INT_COUNT, &dev_count);

        const unsigned int *list;
        unsigned int list_len;
        if (id == PAPI_DEV_TYPE_ID__CUDA && dev_count > 0) {
            for (i = 0; i < dev_count; ++i) {
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_CHAR_DEVICE_NAME, &name);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_CPU_THR_AFFINITY_LIST, &list);
                PAPI_get_dev_attr(handle, i, PAPI_DEV_ATTR__CUDA_UINT_CPU_THR_PER_DEVICE, &list_len);

                if (!quiet) {
                    if (list_len > 0) {
                        printf( "UID                                   : %lu\n", (long unsigned int) i );
                        printf( "Name                                  : %s\n", name );
                        printf( "Affinity                              : " );

                        unsigned int k;
                        for (k = 0; k < list_len; ++k) {
                            printf( "%u ", list[k] );
                        }
                        printf( "\n" );
                    }
                    printf( "\n" );
                }
            }
        }
    }

    if (!quiet) printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        test_pass(__FILE__);
    }

    MPI_Finalize();
    PAPI_shutdown();
    return 0;
}

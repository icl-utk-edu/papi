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

    const PAPI_hw_info_t *info = PAPI_get_hardware_info();

    int dev_type_id, dev_id;

    for (dev_type_id = 0; dev_type_id < PAPI_DEV_TYPE_ID__MAX_NUM; dev_type_id++) {
        if (PAPI_IS_DEV_GPU(NVIDIA, info, dev_type_id)) {
            for (dev_id = 0; dev_id < PAPI_DEV_COUNT(dev_type_id); dev_id++) {
                PAPI_gpu_info_u *dev_info = PAPI_GPU_INFO_STRUCT(info, dev_type_id, dev_id);
                int affinity_count = dev_info->nvidia.affinity.proc_count;
                int proc_id;
                printf( "UID                                   : %lu\n", dev_info->nvidia.uid );
                printf( "Name                                  : %s\n", dev_info->nvidia.name );
                printf( "Affinity                              : " );
                for (proc_id = 0; proc_id < affinity_count; proc_id++) {
                    printf("%d ", dev_info->nvidia.affinity.proc_id_arr[proc_id]);
                }
                printf("\n");
            }
        }

        if (PAPI_IS_DEV_GPU(AMD, info, dev_type_id)) {
            for (dev_id = 0; dev_id < PAPI_DEV_COUNT(dev_type_id); dev_id++) {
                PAPI_gpu_info_u *dev_info = PAPI_GPU_INFO_STRUCT(info, dev_type_id, dev_id);
                int affinity_count = dev_info->amd.affinity.proc_count;
                int proc_id;
                printf( "UID                                   : %lu\n", dev_info->amd.uid );
                printf( "Name                                  : %s\n", dev_info->amd.name );
                printf( "Affinity                              : " );
                for (proc_id = 0; proc_id < affinity_count; proc_id++) {
                    printf("%d ", dev_info->amd.affinity.proc_id_arr[proc_id]);
                }
                printf("\n");
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

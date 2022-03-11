/**
 * @file    query_device_simple.c
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 * test case for sysdetect component
 *
 * @brief
 *  This file contains an example of how to use the sysdetect component to
 *  query NVIDIA GPU device information.
 */
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include "papi_test.h"

static int *
get_num_threads_per_numa( int *numa_affinity, int numas, int threads )
{
    int *threads_per_numa = calloc(numas, sizeof(int));

    int k;
    for (k = 0; k < threads; ++k) {
        threads_per_numa[numa_affinity[k]]++;
    }

    return threads_per_numa;
}

static int **
get_threads_per_numa( int *numa_affinity, int *threads_per_numa, int numas, int threads )
{
    int **numa_threads = malloc(numas * sizeof(*numa_threads));
    int *numa_threads_cnt = calloc(numas, sizeof(*numa_threads_cnt));

    int k;
    for (k = 0; k < numas; ++k) {
        numa_threads[k] = malloc(threads_per_numa[k] * sizeof(int));
    }

    for (k = 0; k < threads; ++k) {
        int node = numa_affinity[k];
        numa_threads[node][numa_threads_cnt[node]++] = k;
    }

    free(numa_threads_cnt);

    return numa_threads;
}

static void
free_numa_threads( int **numa_threads, int numas )
{
    int k;
    for (k = 0; k < numas; ++k) {
        free(numa_threads[k]);
    }
    free(numa_threads);
}

int main(int argc, char *argv[])
{
    int quiet = 0;
    quiet = tests_quiet(argc, argv);

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed\n", retval);
    }

    if (!quiet) {
        printf("Testing sysdetect component with PAPI %d.%d.%d\n",
                PAPI_VERSION_MAJOR(PAPI_VERSION),
                PAPI_VERSION_MINOR(PAPI_VERSION),
                PAPI_VERSION_REVISION(PAPI_VERSION));
    }

    const PAPI_hw_info_t *info = PAPI_get_hardware_info();

    int dev_type_id, dev_id;

    for (dev_type_id = 0; dev_type_id < PAPI_DEV_TYPE_ID__MAX_NUM; dev_type_id++)
    {
        if (PAPI_IS_DEV_CPU(info, dev_type_id))
        {
            for (dev_id = 0; dev_id < PAPI_DEV_COUNT(dev_type_id); dev_id++)
            {
                PAPI_cpu_info_t *dev_info = PAPI_CPU_INFO_STRUCT(info, dev_type_id, dev_id);
                printf( "Id                                    : %d\n", dev_id );
                printf( "Name                                  : %s\n", dev_info->name );
                printf( "CPUID                                 : Family/Model/Stepping %d/%d/%d 0x%02x/0x%02x/0x%02x\n",
                                                                 dev_info->cpuid_family,
                                                                 dev_info->cpuid_model,
                                                                 dev_info->cpuid_stepping,
                                                                 dev_info->cpuid_family,
                                                                 dev_info->cpuid_model,
                                                                 dev_info->cpuid_stepping );
                printf( "Sockets                               : %d\n", dev_info->sockets );
                printf( "Numa regions                          : %d\n", dev_info->numas );
                printf( "Cores per socket                      : %d\n", dev_info->cores );
                printf( "Cores per NUMA region                 : %d\n",
                                                                 (dev_info->threads * dev_info->cores * dev_info->sockets) /
                                                                 dev_info->numas );
                printf( "SMT threads per core                  : %d\n", dev_info->threads );

                if (dev_info->clevel[0].cache[0].type == PAPI_MH_TYPE_INST) {
                    printf( "L1i Cache                             : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                            dev_info->clevel[0].cache[0].size >> 10,
                            dev_info->clevel[0].cache[0].line_size,
                            dev_info->clevel[0].cache[0].num_lines,
                            dev_info->clevel[0].cache[0].associativity );
                } else {
                    printf( "L1d Cache                             : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                            dev_info->clevel[0].cache[0].size >> 10,
                            dev_info->clevel[0].cache[0].line_size,
                            dev_info->clevel[0].cache[0].num_lines,
                            dev_info->clevel[0].cache[0].associativity );
                }

                if (dev_info->clevel[0].cache[1].type == PAPI_MH_TYPE_DATA) {
                    printf( "L1d Cache                             : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                            dev_info->clevel[0].cache[1].size >> 10,
                            dev_info->clevel[0].cache[1].line_size,
                            dev_info->clevel[0].cache[1].num_lines,
                            dev_info->clevel[0].cache[1].associativity );
                } else {
                    printf( "L1i Cache                             : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                            dev_info->clevel[0].cache[1].size >> 10,
                            dev_info->clevel[0].cache[1].line_size,
                            dev_info->clevel[0].cache[1].num_lines,
                            dev_info->clevel[0].cache[1].associativity );
                }

                printf( "L2 Cache                              : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                        dev_info->clevel[1].cache[0].size >> 10,
                        dev_info->clevel[1].cache[0].line_size,
                        dev_info->clevel[1].cache[0].num_lines,
                        dev_info->clevel[1].cache[0].associativity );

                printf( "L3 Cache                              : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                        dev_info->clevel[2].cache[0].size >> 10,
                        dev_info->clevel[2].cache[0].line_size,
                        dev_info->clevel[2].cache[0].num_lines,
                        dev_info->clevel[2].cache[0].associativity );

                int k;
                for (k = 0; k < dev_info->numas; ++k) {
                    printf( "Numa Node %d Memory                    : %d KB\n",
                            k,
                            dev_info->numa_memory[k] );
                }

                int threads = dev_info->threads * dev_info->cores * dev_info->sockets;
                int numas = dev_info->numas;
                int *numa_affinity = dev_info->numa_affinity;
                int *num_threads_per_numa = get_num_threads_per_numa(numa_affinity, numas, threads);
                int **numa_threads = get_threads_per_numa(numa_affinity, num_threads_per_numa, numas, threads);

                for (k = 0; k < numas; ++k) {
                    printf( "Numa Node %d Threads                   : ", k );
                    int l;
                    for (l = 0; l < num_threads_per_numa[k]; ++l) {
                        printf( "%d ", numa_threads[k][l] );
                    }
                    printf( "\n" );
                }

                free_numa_threads(numa_threads, numas);
                free(num_threads_per_numa);

                printf( "\n" );
            }
        }

        if (PAPI_IS_DEV_GPU(NVIDIA, info, dev_type_id))
        {
            for (dev_id = 0; dev_id < PAPI_DEV_COUNT(dev_type_id); dev_id++)
            {
                PAPI_gpu_info_u *dev_info = PAPI_GPU_INFO_STRUCT(info, dev_type_id, dev_id);
                printf( "Id                                    : %d\n", dev_id );
                printf( "UID                                   : %lu\n", dev_info->nvidia.uid );
                printf( "Name                                  : %s\n", dev_info->nvidia.name );
                printf( "Warp size                             : %d\n", dev_info->nvidia.warp_size );
                printf( "Max threads per block                 : %d\n", dev_info->nvidia.max_threads_per_block );
                printf( "Max blocks per multiprocessor         : %d\n", dev_info->nvidia.max_blocks_per_multi_proc );
                printf( "Max shared memory per block           : %d\n", dev_info->nvidia.max_shmmem_per_block );
                printf( "Max shared memory per multiprocessor  : %d\n", dev_info->nvidia.max_shmmem_per_multi_proc );
                printf( "Max block dim x                       : %d\n", dev_info->nvidia.max_block_dim_x );
                printf( "Max block dim y                       : %d\n", dev_info->nvidia.max_block_dim_y );
                printf( "Max block dim z                       : %d\n", dev_info->nvidia.max_block_dim_z );
                printf( "Max grid dim x                        : %d\n", dev_info->nvidia.max_grid_dim_x );
                printf( "Max grid dim y                        : %d\n", dev_info->nvidia.max_grid_dim_y );
                printf( "Max grid dim z                        : %d\n", dev_info->nvidia.max_grid_dim_z );
                printf( "Multiprocessor count                  : %d\n", dev_info->nvidia.multi_processor_count );
                printf( "Multiple kernel per context           : %s\n", dev_info->nvidia.multi_kernel_per_ctx ? "yes" : "no" );
                printf( "Can map host memory                   : %s\n", dev_info->nvidia.can_map_host_mem ? "yes" : "no");
                printf( "Can overlap compute and data transfer : %s\n", dev_info->nvidia.can_overlap_comp_and_data_xfer ? "yes" : "no" );
                printf( "Has unified addressing                : %s\n", dev_info->nvidia.unified_addressing ? "yes" : "no" );
                printf( "Has managed memory                    : %s\n", dev_info->nvidia.managed_memory ? "yes" : "no" );
                printf( "Compute capability                    : %d.%d\n", dev_info->nvidia.major, dev_info->nvidia.minor );

                int k;
                for (k = 0; k < dev_info->nvidia.affinity.proc_count; ++k) {
                    printf( "%d ", dev_info->nvidia.affinity.proc_id_arr[k] );
                }
                printf( "\n" );
                printf( "\n" );
            }
        }

        if (PAPI_IS_DEV_GPU(AMD, info, dev_type_id))
        {
            for (dev_id = 0; dev_id < PAPI_DEV_COUNT(dev_type_id); dev_id++)
            {
                PAPI_gpu_info_u *dev_info = PAPI_GPU_INFO_STRUCT(info, dev_type_id, dev_id);
                printf( "Id                                    : %d\n", dev_id );
                printf( "Name                                  : %s\n", dev_info->amd.name );
                printf( "Wavefront size                        : %u\n", dev_info->amd.wavefront_size );
                printf( "SIMD per compute unit                 : %u\n", dev_info->amd.simd_per_compute_unit );
                printf( "Max threads per workgroup             : %u\n", dev_info->amd.max_threads_per_workgroup );
                printf( "Max waves per compute unit            : %u\n", dev_info->amd.max_waves_per_compute_unit );
                printf( "Max shared memory per workgroup       : %u\n", dev_info->amd.max_shmmem_per_workgroup );
                printf( "Max workgroup dim x                   : %u\n", dev_info->amd.max_workgroup_dim_x );
                printf( "Max workgroup dim y                   : %u\n", dev_info->amd.max_workgroup_dim_y );
                printf( "Max workgroup dim z                   : %u\n", dev_info->amd.max_workgroup_dim_z );
                printf( "Max grid dim x                        : %u\n", dev_info->amd.max_grid_dim_x );
                printf( "Max grid dim y                        : %u\n", dev_info->amd.max_grid_dim_y );
                printf( "Max grid dim z                        : %u\n", dev_info->amd.max_grid_dim_z );
                printf( "Compute unit count                    : %u\n", dev_info->amd.compute_unit_count );
                printf( "Compute capability                    : %u.%u\n", dev_info->amd.major, dev_info->amd.minor );
                printf( "\n" );
            }
        }
    }

    PAPI_shutdown();

    if (!quiet) printf("\n");
    test_pass(__FILE__);

    return 0;
}

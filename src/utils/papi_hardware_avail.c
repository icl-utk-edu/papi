/** file papi_hardware_avail.c
  * @page papi_hardware_avail
  * @brief papi_hardware_avail utility.
  * @section  NAME
  *     papi_hardware_avail - provides detailed information on the hardware available in the system.
  *
  * @section Synopsis
  *
  * @section Description
  *     papi_hardware_avail is a PAPI utility program that reports information
  *     about the hardware devices equipped in the system.
  *
  * @section Options
  *      <ul>
  *     <li>-h help message
  *      </ul>
  *
  * @section Bugs
  *     There are no known bugs in this utility.
  *     If you find a bug, it should be reported to the
  *     PAPI Mailing List at <ptools-perfapi@icl.utk.edu>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi.h"
#include "print_header.h"

typedef struct command_flags
{
    int help;
} command_flags_t;

static void
print_help( char **argv )
{
    printf( "This is the PAPI hardware avail program.\n" );
    printf( "It provides availability of system's equipped hardware devices.\n" );
    printf( "Usage: %s [options]\n", argv[0] );
    printf( "Options:\n\n" );
    printf( "  --help, -h    print this help message\n" );
}

static void
parse_args( int argc, char **argv, command_flags_t * f )
{
    int i;

    /* Look for all currently defined commands */
    memset( f, 0, sizeof ( command_flags_t ) );
    for ( i = 1; i < argc; i++ ) {
        if ( !strcmp( argv[i], "-h" ) || !strcmp( argv[i], "--help" ) )
            f->help = 1;
        else
            printf( "%s is not supported\n", argv[i] );
    }

    /* if help requested, print and bail */
    if ( f->help ) {
        print_help( argv );
        exit( 1 );
    }

}

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

int
main( int argc, char **argv )
{
    int i;
    int retval;
    const PAPI_hw_info_t *hwinfo = NULL;
    const PAPI_component_info_t *cmpinfo = NULL;
    command_flags_t flags;
    int num_dev_types, numcmp;
    int sysdetect_avail = 0;

    /* Initialize before parsing the input arguments */
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if ( retval != PAPI_VER_CURRENT ) {
        fprintf(stderr,"Error!  PAPI_library_init\n");
        return retval;
    }

    parse_args( argc, argv, &flags );

    retval = PAPI_set_debug( PAPI_VERB_ECONT );
    if ( retval != PAPI_OK ) {
        fprintf(stderr,"Error!  PAPI_set_debug\n");
        return retval;
    }

    hwinfo = PAPI_get_hardware_info();

    numcmp = PAPI_num_components( );
    for (i = 0; i < numcmp; i++) {
        cmpinfo = PAPI_get_component_info( i );
        if (strcmp("sysdetect", cmpinfo->name) == 0)
            sysdetect_avail = 1;
    }

    if (sysdetect_avail == 0) {
        fprintf(stderr, "Error! Sysdetect component not enabled\n");
        return 0;
    }

    /* All possible hardware devices detectable */
    num_dev_types = PAPI_DEV_TYPE_ID__MAX_NUM;

    PAPI_dev_type_info_t *dev_type_info = hwinfo->dev_type_arr;

    printf( "\nDevice Summary -----------------------------------------------------------------\n" );
    printf( "Vendor           DevCount \n" );
    for ( i = 0; i < num_dev_types; i++ ) {
        printf( "%-18s (%d)\n",
                dev_type_info[i].vendor,
                dev_type_info[i].num_devices);
        printf( " \\-> Status: %s\n", dev_type_info[i].status );
    }

    printf( "\n" );
    printf( "\nDevice Information -------------------------------------------------------------\n" );
    for ( i = 0; i < num_dev_types; i++ ) {

        int j;

        if ( dev_type_info[i].id == PAPI_DEV_TYPE_ID__CPU && dev_type_info[i].num_devices > 0 ) {
            printf( "Vendor                                : %s\n", dev_type_info[i].vendor );

            PAPI_cpu_info_t *cpu_info = (PAPI_cpu_info_t *) dev_type_info[i].dev_info_arr;
            for ( j = 0; j < dev_type_info[i].num_devices; j++ ) {
                printf( "Id                                    : %d\n", j );
                printf( "Name                                  : %s\n", cpu_info[j].name );
                printf( "CPUID                                 : Family/Model/Stepping %d/%d/%d 0x%02x/0x%02x/0x%02x\n",
                                                                 cpu_info[j].cpuid_family,
                                                                 cpu_info[j].cpuid_model,
                                                                 cpu_info[j].cpuid_stepping,
                                                                 cpu_info[j].cpuid_family,
                                                                 cpu_info[j].cpuid_model,
                                                                 cpu_info[j].cpuid_stepping );
                printf( "Sockets                               : %d\n", cpu_info[j].sockets );
                printf( "Numa regions                          : %d\n", cpu_info[j].numas );
                printf( "Cores per socket                      : %d\n", cpu_info[j].cores );
                printf( "Cores per NUMA region                 : %d\n",
                                                                 (cpu_info[j].threads * cpu_info[j].cores * cpu_info[j].sockets) /
                                                                 cpu_info[j].numas );
                printf( "SMT threads per core                  : %d\n", cpu_info[j].threads );

                if (cpu_info[j].clevel[0].cache[0].type == PAPI_MH_TYPE_INST) {
                    printf( "L1i Cache                             : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                            cpu_info[j].clevel[0].cache[0].size >> 10,
                            cpu_info[j].clevel[0].cache[0].line_size,
                            cpu_info[j].clevel[0].cache[0].num_lines,
                            cpu_info[j].clevel[0].cache[0].associativity );
                } else {
                    printf( "L1d Cache                             : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                            cpu_info[j].clevel[0].cache[0].size >> 10,
                            cpu_info[j].clevel[0].cache[0].line_size,
                            cpu_info[j].clevel[0].cache[0].num_lines,
                            cpu_info[j].clevel[0].cache[0].associativity );
                }

                if (cpu_info[j].clevel[0].cache[1].type == PAPI_MH_TYPE_DATA) {
                    printf( "L1d Cache                             : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                            cpu_info[j].clevel[0].cache[1].size >> 10,
                            cpu_info[j].clevel[0].cache[1].line_size,
                            cpu_info[j].clevel[0].cache[1].num_lines,
                            cpu_info[j].clevel[0].cache[1].associativity );
                } else {
                    printf( "L1i Cache                             : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                            cpu_info[j].clevel[0].cache[1].size >> 10,
                            cpu_info[j].clevel[0].cache[1].line_size,
                            cpu_info[j].clevel[0].cache[1].num_lines,
                            cpu_info[j].clevel[0].cache[1].associativity );
                }

                printf( "L2 Cache                              : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                        cpu_info[j].clevel[1].cache[0].size >> 10,
                        cpu_info[j].clevel[1].cache[0].line_size,
                        cpu_info[j].clevel[1].cache[0].num_lines,
                        cpu_info[j].clevel[1].cache[0].associativity );

                printf( "L3 Cache                              : Size/LineSize/Lines/Assoc %dKB/%dB/%d/%d\n",
                        cpu_info[j].clevel[2].cache[0].size >> 10,
                        cpu_info[j].clevel[2].cache[0].line_size,
                        cpu_info[j].clevel[2].cache[0].num_lines,
                        cpu_info[j].clevel[2].cache[0].associativity );

                int k;
                for (k = 0; k < cpu_info[j].numas; ++k) {
                    printf( "Numa Node %d Memory                    : %d KB\n",
                            k,
                            cpu_info[j].numa_memory[k] );
                }

                int threads = cpu_info[j].threads * cpu_info[j].cores * cpu_info[j].sockets;
                int numas = cpu_info[j].numas;
                int *numa_affinity = cpu_info[j].numa_affinity;
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

        if ( dev_type_info[i].id == PAPI_DEV_TYPE_ID__NVIDIA_GPU && dev_type_info[i].num_devices > 0 ) {
            printf( "Vendor                                : %s\n", dev_type_info[i].vendor );

            PAPI_gpu_info_u *dev_info = (PAPI_gpu_info_u *) dev_type_info[i].dev_info_arr;
            for ( j = 0; j < dev_type_info[i].num_devices; j++ ) {
                printf( "Id                                    : %d\n", j );
                printf( "UID                                   : %lu\n", dev_info[j].nvidia.uid );
                printf( "Name                                  : %s\n", dev_info[j].nvidia.name );
                printf( "Warp size                             : %d\n", dev_info[j].nvidia.warp_size );
                printf( "Max threads per block                 : %d\n", dev_info[j].nvidia.max_threads_per_block );
                printf( "Max blocks per multiprocessor         : %d\n", dev_info[j].nvidia.max_blocks_per_multi_proc );
                printf( "Max shared memory per block           : %d\n", dev_info[j].nvidia.max_shmmem_per_block );
                printf( "Max shared memory per multiprocessor  : %d\n", dev_info[j].nvidia.max_shmmem_per_multi_proc );
                printf( "Max block dim x                       : %d\n", dev_info[j].nvidia.max_block_dim_x );
                printf( "Max block dim y                       : %d\n", dev_info[j].nvidia.max_block_dim_y );
                printf( "Max block dim z                       : %d\n", dev_info[j].nvidia.max_block_dim_z );
                printf( "Max grid dim x                        : %d\n", dev_info[j].nvidia.max_grid_dim_x );
                printf( "Max grid dim y                        : %d\n", dev_info[j].nvidia.max_grid_dim_y );
                printf( "Max grid dim z                        : %d\n", dev_info[j].nvidia.max_grid_dim_z );
                printf( "Multiprocessor count                  : %d\n", dev_info[j].nvidia.multi_processor_count );
                printf( "Multiple kernels per context          : %s\n", dev_info[j].nvidia.multi_kernel_per_ctx ? "yes" : "no" );
                printf( "Can map host memory                   : %s\n", dev_info[j].nvidia.can_map_host_mem ? "yes" : "no");
                printf( "Can overlap compute and data transfer : %s\n", dev_info[j].nvidia.can_overlap_comp_and_data_xfer ? "yes" : "no" );
                printf( "Has unified addressing                : %s\n", dev_info[j].nvidia.unified_addressing ? "yes" : "no" );
                printf( "Has managed memory                    : %s\n", dev_info[j].nvidia.managed_memory ? "yes" : "no" );
                printf( "Compute capability                    : %d.%d\n", dev_info[j].nvidia.major, dev_info[j].nvidia.minor );

                if (dev_info[j].nvidia.affinity.proc_count > 0) {
                    printf( "Affinity                              : ");
                    int k;
                    for (k = 0; k < dev_info[j].nvidia.affinity.proc_count; ++k) {
                        printf( "%d ", dev_info[j].nvidia.affinity.proc_id_arr[k] );
                    }
                    printf( "\n" );
                }
                printf( "\n" );
            }
        }

        if ( dev_type_info[i].id == PAPI_DEV_TYPE_ID__AMD_GPU && dev_type_info[i].num_devices > 0 ) {
            printf( "Vendor                                : %s\n", dev_type_info[i].vendor );

            PAPI_gpu_info_u *dev_info = (PAPI_gpu_info_u *) dev_type_info[i].dev_info_arr;
            for ( j = 0; j < dev_type_info[i].num_devices; j++ ) {
                printf( "Id                                    : %d\n", j );
                printf( "Name                                  : %s\n", dev_info[j].amd.name );
                printf( "Wavefront size                        : %u\n", dev_info[j].amd.wavefront_size );
                printf( "SIMD per compute unit                 : %u\n", dev_info[j].amd.simd_per_compute_unit );
                printf( "Max threads per workgroup             : %u\n", dev_info[j].amd.max_threads_per_workgroup );
                printf( "Max waves per compute unit            : %u\n", dev_info[j].amd.max_waves_per_compute_unit );
                printf( "Max shared memory per workgroup       : %u\n", dev_info[j].amd.max_shmmem_per_workgroup );
                printf( "Max workgroup dim x                   : %u\n", dev_info[j].amd.max_workgroup_dim_x );
                printf( "Max workgroup dim y                   : %u\n", dev_info[j].amd.max_workgroup_dim_y );
                printf( "Max workgroup dim z                   : %u\n", dev_info[j].amd.max_workgroup_dim_z );
                printf( "Max grid dim x                        : %u\n", dev_info[j].amd.max_grid_dim_x );
                printf( "Max grid dim y                        : %u\n", dev_info[j].amd.max_grid_dim_y );
                printf( "Max grid dim z                        : %u\n", dev_info[j].amd.max_grid_dim_z );
                printf( "Compute unit count                    : %u\n", dev_info[j].amd.compute_unit_count );
                printf( "Compute capability                    : %u.%u\n", dev_info[j].amd.major, dev_info[j].amd.minor );
                printf( "\n" );
            }
        }
    }

    printf( "--------------------------------------------------------------------------------\n" );

    PAPI_shutdown();
    return 0;
}

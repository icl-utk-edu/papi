# SYSDETECT Component

The SYSDETECT component allows PAPI users to query comprehensive system
information. The information is gathered at PAPI_library_init() time and
presented to the user through appropriate APIs. The component works
similarly to other components, which means that hardware information for
a specific device might not be available at runtime if, e.g., the device
runtime software is not installed.

At the moment the infrastructure defines the following device types:

  - PAPI_DEV_TYPE_ID__CPU        : for all CPU devices from any vendor
  - PAPI_DEV_TYPE_ID__NVIDIA_GPU : for all GPU devices from NVIDIA
  - PAPI_DEV_TYPE_ID__AMD_GPU    : for all GPU devices from AMD

These types are defined as PAPI_dev_type_id_e enums.

Every device is scanned to gather information when the component is
initialized. If there is no installed hardware for the considered
device type, or the matching runtime library, is not found the
corresponding information is filled with zeros and a status string
filled with the message describing why this happened.

The low-level API provides the following structures for querying device
information:

    - PAPI_gpu_info_u
      {
          struct {
              char name[PAPI_2MAX_STR_LEN];
              int warp_size;
              int max_threads_per_block;
              int max_blocks_per_multi_proc;
              int max_shmmem_per_block;
              int max_shmmem_per_multi_proc;
              int max_block_dim_x;
              int max_block_dim_y;
              int max_block_dim_z;
              int multi_processor_count;
              int multi_kernel_per_ctx;
              int can_map_host_mem;
              int can_overlap_comp_and_datatx;
              int unified_memory;
              int managed_memory;
              int major;
              int minor;
          } nvidia;

          struct  {
              ...
          } amd;
      };

      PAPI_cpu_info_t {
          ...
      };

    - The above is available through the PAPI_hw_info_t structure that
      has been extended with the (PAPI_dev_type_info_t) dev_type_arr
      field. The structure of this field contains the following info:

      PAPI_dev_type_info_t
      {
          PAPI_dev_type_id_e id;
          char vendor[PAPI_MAX_STR_LEN];
          char status[PAPI_MAX_STR_LEN];
          int num_devices;
          PAPI_dev_info_u *dev_info_arr;
      };

      Clients can get a pointer to PAPI_hw_info_t using
      PAPI_get_hardware_info(). Afterwards, they can access directly the
      dev_type_arr array to learn about what devices are available in
      the system, the vendor name and their number. If clients want to
      access detailed information for a particular device they can do so
      through the dev_info_arr field in PAPI_dev_type_info_t.

* [Enabling the SYSDETECT Component](#markdown-header-enabling-the-sysdetect-component)

## Enabling the SYSDETECT Component

To enable reading of SYSDETECT counters the user needs to link against a
PAPI library that was configured with the SYSDETECT component enabled.  As an
example the following command: `./configure --with-components="sysdetect"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

The utility program papi_hardware_avail uses the SYSDETECT component to report
installed and configured hardware information to the command line.

/**
 * @file amdsmi_check_for_supported_metrics.c
 *
 * @brief This vendor sample code will initialize the amd_smi library, retrieve the socket handles,
 *        and then for each socket handle obtain the processor handles. Each processor handle will then
 *        have amdsmi_get_gpu_metrics_info called on it to check for support of amd_smi metrics (non-exhaustive).
 * 
 *        This file also serves as a starting point for a reproducer if a bug report needs to be
 *        sent to AMD for the amd_smi component.
 *
 *        Tested on:
 *        - Odyssey (4 * MI300As) at Oregon with ROCm 7.2.4.
 */

// C STL headers.
#include <stdio.h>
#include <stdlib.h>

// ROCm headers.
#include <amdsmi.h>

int main()
{
    uint64_t flags = AMDSMI_INIT_AMD_GPUS;
    amdsmi_status_t status = amdsmi_init(flags);
    if (status != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "Line %d: Call to amdsmi_init failed with error code %d.\n", __LINE__, status);
        exit(EXIT_FAILURE);
    } 

    uint32_t socket_count = 0;
    status = amdsmi_get_socket_handles(&socket_count, NULL);
    if (status != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "Line %d: Call to amdsmi_get_socket_handles failed with error code %d.\n", __LINE__, status);
        exit(EXIT_FAILURE);
    }

    amdsmi_socket_handle *socket_handles = (amdsmi_socket_handle *) malloc(socket_count * sizeof(amdsmi_socket_handle));
    if (socket_handles == NULL) {
        fprintf(stderr, "Line %d: Failed to allocate memory for socket_handles.\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    status = amdsmi_get_socket_handles(&socket_count, socket_handles);
    if (status != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "Line %d: Call to amdsmi_get_socket_handles failed with error code %d.\n", __LINE__, status);
        exit(EXIT_FAILURE);
    }

    for (uint32_t socket = 0; socket < socket_count; socket++) {
        uint32_t processor_count = 0;
        status = amdsmi_get_processor_handles(socket_handles[socket], &processor_count, NULL);
        if (status != AMDSMI_STATUS_SUCCESS) {
            fprintf(stderr, "Line %d: Call to amdsmi_get_processor_handles failed with error code %d.\n", __LINE__, status);
            exit(EXIT_FAILURE);
        }

        amdsmi_processor_handle *processor_handles = (amdsmi_processor_handle *) malloc(processor_count * sizeof(amdsmi_processor_handle));
        if (processor_handles == NULL) {
            fprintf(stderr, "Line %d: Failed to allocate memory for processor_handles.\n", __LINE__);
            exit(EXIT_FAILURE);
        }

        status = amdsmi_get_processor_handles(socket_handles[socket], &processor_count, processor_handles);
        if (status != AMDSMI_STATUS_SUCCESS) {
            fprintf(stderr, "Line %d: Call to amdsmi_get_processor_handles failed with error code %d.\n", __LINE__, status);
            exit(EXIT_FAILURE);
        }

        for (uint32_t processor = 0; processor < processor_count; processor++) {
            amdsmi_asic_info_t asic;
            status = amdsmi_get_gpu_asic_info(processor_handles[processor], &asic);
            if (status != AMDSMI_STATUS_SUCCESS) {
                fprintf(stderr, "Line %d: Call to amdsmi_get_gpu_asic_info failed with error code %d.\n", __LINE__, status);
                exit(EXIT_FAILURE);
            }

            amdsmi_gpu_metrics_t gpu_metrics;
            status = amdsmi_get_gpu_metrics_info(processor_handles[processor], &gpu_metrics);
            if (status != AMDSMI_STATUS_SUCCESS) {
                fprintf(stderr, "Line %d: Call to amdsmi_get_gpu_metrics_info failed with error code %d.\n", __LINE__, status);
                exit(EXIT_FAILURE);
            }
            if (socket != 0 && socket < socket_count) {
                fprintf(stdout, "\n");
            }
            printf("GPU: %s, Format Revision: %u, Context Revision: %u\n", asic.market_name, gpu_metrics.common_header.format_revision, gpu_metrics.common_header.content_revision);

            // Subset of metric checks for v1.0.
            if (gpu_metrics.temperature_vrgfx != UINT16_MAX) {
                fprintf(stdout, "Available: temperature_vrgfx\n");
            }
            else {
                fprintf(stdout, "Not availabile: temperature_vrgfx\n");
            }

            if (gpu_metrics.temperature_vrsoc != UINT16_MAX) {
                fprintf(stdout, "Available: temperature_vrsoc\n");
            }
            else {
                fprintf(stdout, "Not available: temperature_vrsoc\n");
            }

            if (gpu_metrics.average_vclk1_frequency != UINT16_MAX) {
                fprintf(stdout, "Available: average_vclk1_frequency\n");
            }
            else {
                fprintf(stdout, "Not Availabile: average_vclk1_frequency\n");
            }

            if (gpu_metrics.average_dclk1_frequency != UINT16_MAX) {
                fprintf(stdout, "Available: average_dclk1_frequency\n");
            }
            else {
                fprintf(stdout, "Not Availabile: average_dclk1_frequency\n");
            }

            if (gpu_metrics.pcie_link_width != UINT16_MAX) {
                fprintf(stdout, "Available: pcie_link_width\n");
            }   
            else {
                fprintf(stdout, "Not available: pcie_link_width\n");
            }

            if (gpu_metrics.pcie_link_speed != UINT16_MAX) {
                fprintf(stdout, "Available: pcie_link_speed\n");
            }   
            else {
                fprintf(stdout, "Not available: pcie_link_speed\n");
            }

            // Subset of metric checks for v1.1.
            if (gpu_metrics.gfx_activity_acc != UINT32_MAX) {
                fprintf(stdout, "Availabile: gfx_activity_acc\n");
            }
            else {
                fprintf(stdout, "Not availabile: gfx_activity_acc\n");
            }

            // Subset of metric checks for v1.2.
            if (gpu_metrics.firmware_timestamp != UINT64_MAX) {
                fprintf(stdout, "Availabile: firmware_timestamp\n");
            }
            else {
                fprintf(stdout, "Not availabile: firmware_timestamp\n");
            } 

            // Subset of metric checks for v1.3.
            if (gpu_metrics.voltage_soc != UINT16_MAX) {
                fprintf(stdout, "Availabile: voltage_soc\n");
            }
            else {
                fprintf(stdout, "Not availabile: voltage_soc\n");
            }

            // Subset of metric checks for v1.4.
            if (gpu_metrics.pcie_bandwidth_acc != UINT64_MAX) {
                fprintf(stdout, "Availabile: pcie_bandwidth_acc\n");
            }
            else {
                fprintf(stdout, "Not availabile: pcie_bandwidth_acc\n");
            }

            if (gpu_metrics.pcie_bandwidth_inst != UINT64_MAX) {
                fprintf(stdout, "Availabile: pcie_bandwidth_inst\n");
            }
            else {
                fprintf(stdout, "Not availabile: pcie_bandwidth_inst\n");
            } 

            if (gpu_metrics.pcie_l0_to_recov_count_acc != UINT64_MAX) {
                fprintf(stdout, "Availabile: pcie_l0_to_recov_count_acc\n");
            }
            else {
                fprintf(stdout, "Not availabile: pcie_l0_to_recov_count_acc\n");
            }

            if (gpu_metrics.pcie_replay_count_acc != UINT64_MAX) {
                fprintf(stdout, "Availabile: pcie_replay_count_acc\n");
            }
            else {
                fprintf(stdout, "Not availabile: pcie_replay_count_acc\n");
            } 

            if (gpu_metrics.pcie_replay_rover_count_acc != UINT64_MAX) {
                fprintf(stdout, "Availabile: pcie_replay_rover_count_acc\n");
            }
            else {
                fprintf(stdout, "Not availabile: pcie_replay_rover_count_acc\n");
            }

            unsigned int read_data_acc_count = 0;
            for (int links = 0; links < AMDSMI_MAX_NUM_XGMI_LINKS; links++) {
                if (gpu_metrics.xgmi_read_data_acc[links] != UINT64_MAX) {
                    read_data_acc_count++;
                }
            }
            if (read_data_acc_count > 0) {
                fprintf(stdout, "Available: xgmi_read_data_acc\n");
            }
            else {
                fprintf(stdout, "Not available: xgmi_read_data_acc\n");
            }

            unsigned int write_data_acc_count = 0;
            for (int links = 0; links < AMDSMI_MAX_NUM_XGMI_LINKS; links++) {
                if (gpu_metrics.xgmi_write_data_acc[links] != UINT64_MAX) {
                    write_data_acc_count++;
                }
            }
            if (write_data_acc_count > 0) {
                fprintf(stdout, "Available: xgmi_write_data_acc\n");
            }
            else {
                fprintf(stdout, "Not available: xgmi_write_data_acc\n");
            }
            
            // Subset of metric checks for v1.5.
            if (gpu_metrics.pcie_nak_sent_count_acc != UINT32_MAX) {
                fprintf(stdout, "Availabile: pcie_nak_sent_count_acc\n");
            }
            else {
                fprintf(stdout, "Not availabile: pcie_nak_sent_count_acc\n");
            }

            unsigned int jpeg_activity_count = 0;
            for (int jpeg = 0; jpeg < AMDSMI_MAX_NUM_JPEG; jpeg++) {
                if (gpu_metrics.jpeg_activity[jpeg] != UINT16_MAX) {
                    jpeg_activity_count++; 
                }
            }
            if (jpeg_activity_count > 0) {
                fprintf(stdout, "Available: jpeg_activity_count\n");
            }
            else {
                fprintf(stdout, "Not available: jpeg_activity_count\n");
            }

            // Subset of metric checks for v1.6.
            if (gpu_metrics.accumulation_counter != UINT64_MAX) {
                fprintf(stdout, "Available: accumulation_counter\n");
            }
            else {
                fprintf(stdout, "Not availabile: accumulation_counter\n");
            }

            if (gpu_metrics.num_partition != UINT16_MAX) {
                fprintf(stdout, "Available: num_partition\n");
            }
            else {
                fprintf(stdout, "Not availabile: num_partition\n");
            }

            unsigned int host_limit_acc_count = 0;
            for (int xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; xcp++) {
                for (int xcc = 0; xcc < AMDSMI_MAX_NUM_XCC; xcc++) {
                    if (gpu_metrics.xcp_stats[xcp].gfx_below_host_limit_acc[xcc] != UINT64_MAX) {
                        host_limit_acc_count++;
                    }
                }
            }
            if (host_limit_acc_count > 0) {
                fprintf(stdout, "Available: gfx_below_host_limit_acc\n");
            }
            else {
                fprintf(stdout, "Not available: gfx_below_host_limit_acc\n");
            }

            // Subset of metric checks for v1.7.
            if (gpu_metrics.vram_max_bandwidth != UINT64_MAX) {
                fprintf(stdout, "Available: vram_max_bandwidth\n");
            }
            else {
                fprintf(stdout, "Not availabile: vram_max_bandwidth\n");
            }
        }
    }

    return 0;
}

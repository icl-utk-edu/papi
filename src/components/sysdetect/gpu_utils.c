#include <errno.h>
#include <dirent.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "papi.h"
#include "papi_debug.h"

int query_available_vendor_devices(const char *vendor_pci_id, int *vendor_device_count)
{
    // NOTE: Contents of /sys/bus/pci/devices are symlinked from /sys/devices.
    const char *pci_devices_path = "/sys/bus/pci/devices";
    DIR *dir = opendir(pci_devices_path);
    if (dir == NULL) {
        SUBDBG("Failed to open the directory: %s. Reason given: %s.\n", pci_devices_path, strerror(errno));
        return PAPI_ESYS;
    }   

    int status = 0;
    struct dirent *dir_entry = NULL;
    while ((dir_entry = readdir(dir)) != NULL) {
        const char *current_directory = ".", *parent_directory = "..";
        if (strcmp(dir_entry->d_name, current_directory) == 0 ||
            strcmp(dir_entry->d_name, parent_directory) == 0) {
            SUBDBG("Detected current or parent directories. Continuing.\n");
            continue;
        }   

        char pci_class_path[PATH_MAX] = { 0 };
        int str_len = snprintf(pci_class_path, sizeof(pci_class_path), "%s/%s/class", pci_devices_path, dir_entry->d_name);
        if (str_len < 0 || (size_t) str_len >= sizeof(pci_class_path)) {
            SUBDBG("Failed to write path to the class file. Continuing.\n");
            continue;
        }   
    
        const char *mode = "r";
        FILE *pci_class_file = fopen(pci_class_path, mode);
        if (pci_class_file == NULL) {
            SUBDBG("Failed to open the file: %s. Reason given: %s. Continuing.\n", pci_class_path, strerror(errno));
            continue;
        }

        char class_buf[PAPI_MIN_STR_LEN] = { 0 };
        if (fgets(class_buf, sizeof(class_buf), pci_class_file) == NULL) {
            SUBDBG("Failed to read bytes from the file: %s. Reason given: %s. Continuing\n", pci_class_path, strerror(errno));
            goto close_class_file;
        }

        const char *pci_base_class_display = "0x03"; 
        if (strncmp(class_buf, pci_base_class_display, strlen(pci_base_class_display)) == 0) {
            char pci_vendor_path[PATH_MAX] = { 0 };
            str_len = snprintf(pci_vendor_path, sizeof(pci_vendor_path), "%s/%s/vendor", pci_devices_path, dir_entry->d_name);
            if (str_len < 0 || (size_t) str_len >= sizeof(pci_vendor_path)) {
                SUBDBG("Failed to write path to the vendor file. Continuing.\n");
                goto close_class_file;
            }   

            FILE *pci_vendor_file = fopen(pci_vendor_path, mode);
            if (pci_vendor_file == NULL) {
                SUBDBG("Failed to open the file: %s. Reason given: %s.\n", pci_vendor_path, strerror(errno));
                goto close_class_file; 
            }

            // NOTE: fgets will include the newline as a character.
            char vendor_buf[PAPI_MIN_STR_LEN] = { 0 };
            if (fgets(vendor_buf, sizeof(vendor_buf), pci_vendor_file) == NULL) {
                SUBDBG("Failed to read bytes from the file: %s. Reason given: %s.\n", pci_vendor_path, strerror(errno));
                goto close_vendor_file;
            }

            if (strncmp(vendor_buf, vendor_pci_id, strlen(vendor_pci_id)) == 0) {
                (*vendor_device_count)++;
            }

            close_vendor_file:
            status = fclose(pci_vendor_file);
            if (status == EOF) {
                SUBDBG("Failed to close the file: %s. Reason given: %s."
                       " Proceeding.\n", pci_vendor_path, strerror(errno));
            }
        }

        close_class_file: 
        status = fclose(pci_class_file);
        if (status == EOF) {
            SUBDBG("Failed to close the file: %s. Reason given: %s."
                   " Proceeding.\n", pci_class_path, strerror(errno)); 
        }
    }

    status = closedir(dir);
    if (status == -1) {
        SUBDBG("Failed to close the directory: %s. Reason given: %s."
               " Proceeding.\n", pci_devices_path, strerror(errno));
    }   

    return PAPI_OK;
}

void set_dev_type_info_status(const char *message, size_t size, char *status)
{
    int count = snprintf(status, size, "%s", message);
    if (count < 0 || (size_t) count >= size) {
        SUBDBG("The status string will be truncated. Continuing.\n");
    }   
}

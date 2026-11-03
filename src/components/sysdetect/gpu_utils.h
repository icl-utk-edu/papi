#ifndef _GPU_UTILS_H
#define _GPU_UTILS_H

int query_available_vendor_devices(const char *vendor_pci_id, int *vendor_device_count);
void set_dev_type_info_status(const char *message, size_t size, char *status);

#endif // _GPU_UTILS_H

# Set ROCM_PATH only if it is not set already.
ROCM_PATH ?= /opt/rocm
# Define includes needed for vendor reproducers.
AMDSMI_VENDOR_REPRODUCERS_INCLUDES += -I$(ROCM_PATH)/include/amd_smi
# Define libraries needed for vendor reproducers.
AMDSMI_VENDOR_REPRODUCERS_LIBS += -L$(ROCM_PATH)/lib -lamd_smi
# Define the path to the amdclang compiler.
amdclang_amdsmi_vendor_reproducers := $(ROCM_PATH)/bin/amdclang

amdsmi_vendor_reproducers := amdsmi_check_for_supported_metrics
amdsmi_vendor_reproducers_all: $(amdsmi_vendor_reproducers)

amdsmi_check_for_supported_metrics: amdsmi_check_for_supported_metrics.c
	$(amdclang_amdsmi_vendor_reproducers) $(AMDSMI_VENDOR_REPRODUCERS_INCLUDES) $< -o $@ $(AMDSMI_VENDOR_REPRODUCERS_LIBS)

clean:
	rm -rf $(amdsmi_vendor_reproducers)

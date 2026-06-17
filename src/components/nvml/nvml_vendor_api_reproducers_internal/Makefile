# Set CUDA_HOME only if it is not set already
CUDA_HOME ?= /usr/local/cuda
# Define includes needed for vendor reproducers
NVML_VENDOR_REPRODUCERS_INCLUDES += -I$(CUDA_HOME)/include
# Define libraries needed for vendor reproducers
NVML_VENDOR_REPRODUCERS_LIBS += -L$(CUDA_HOME)/lib64 -lnvidia-ml
# Define flags needed for vendor reproducers
NVML_VENDOR_REPRODUCER_FLAGS += -Wno-deprecated-gpu-targets
# Define the path to the NVIDIA CUDA Compiler
nvcc_nvml_vendor_reproducers := $(CUDA_HOME)/bin/nvcc

nvml_vendor_reproducers := nvml_get_device_field_vendor_reproducer
nvml_vendor_reproducers_all: $(nvml_vendor_reproducers)

nvml_get_device_field_vendor_reproducer: nvml_get_device_field_vendor_reproducer.cu
	$(nvcc_nvml_vendor_reproducers) $(NVML_VENDOR_REPRODUCER_FLAGS) $(NVML_VENDOR_REPRODUCERS_INCLUDES) nvml_get_device_field_vendor_reproducer.cu -o nvml_get_device_field_vendor_reproducer $(NVML_VENDOR_REPRODUCERS_LIBS)
clean:
	rm -rf $(nvml_vendor_reproducers)

# $Id$
# Makefile for the Performance-monitoring counters driver.
#

O_TARGET := kperfctr.o
driver-objs-y := init.o
driver-objs-$(CONFIG_X86) += x86.o
tests-objs-$(CONFIG_X86) := x86_tests.o
kernel-objs-$(CONFIG_X86) := x86_setup.o
driver-objs-$(CONFIG_PERFCTR_INIT_TESTS) += $(tests-objs-y)
driver-objs-$(CONFIG_PERFCTR_VIRTUAL) += virtual.o
stub-objs-$(CONFIG_PERFCTR)-$(CONFIG_PERFCTR_VIRTUAL) := virtual_stub.o
driver-objs-$(CONFIG_PERFCTR_GLOBAL) += global.o
m-objs-$(CONFIG_PERFCTR) := perfctr.o
y-objs-$(CONFIG_PERFCTR) := $(driver-objs-y)
kernel-objs-y += $(stub-objs-m-y)

export-objs := $(kernel-objs-y)
obj-y := $(kernel-objs-y) $(y-objs-y)
obj-m := $(m-objs-m)
list-multi := perfctr.o
perfctr-objs := $(driver-objs-y)

ifeq ($(VERSION).$(PATCHLEVEL),2.2)
multi-y		:= $(filter $(list-multi), $(obj-y))
multi-m		:= $(filter $(list-multi), $(obj-m))
int-y		:= $(sort $(foreach m, $(multi-y), $($(basename $(m))-objs)))
int-m		:= $(sort $(foreach m, $(multi-m), $($(basename $(m))-objs)))
obj-m		:= $(filter-out $(obj-y), $(obj-m))
int-m		:= $(filter-out $(int-y), $(int-m))
O_OBJS		:= $(sort $(filter-out $(export-objs), $(obj-y)))
OX_OBJS		:= $(sort $(filter     $(export-objs), $(obj-y)))
M_OBJS		:= $(sort $(filter-out $(export-objs), $(obj-m)))
MX_OBJS		:= $(sort $(filter     $(export-objs), $(obj-m)))
MI_OBJS		:= $(sort $(filter-out $(export-objs), $(int-m)))
MIX_OBJS	:= $(sort $(filter     $(export-objs), $(int-m)))
endif

include $(TOPDIR)/Rules.make

perfctr.o:	$(perfctr-objs)
	$(LD) -r -o $@ $(perfctr-objs)

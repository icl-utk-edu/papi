# $Id$
# Makefile for the Performance-monitoring counters driver.

# construct various object file lists:
# kernel-objs-y		kernel objects exporting symbols
# y-objs-y		kernel objects not exporting symbols
# m-objs-m		perfctr.o if driver is module, empty otherwise
# driver-objs-y		objects for perfctr.o module, or empty
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

export-objs		:= $(kernel-objs-y)
perfctr-objs		:= $(driver-objs-y)
obj-m			+= $(m-objs-m)

ifeq ($(VERSION).$(PATCHLEVEL),2.5)
# for 2.5.19 or later
obj-$(CONFIG_KPERFCTR)	+= kperfctr.o
kperfctr-objs		:= $(kernel-objs-y) $(y-objs-y)
else
# for 2.4 and 2.2
O_TARGET		:= kperfctr.o
obj-y			:= $(kernel-objs-y) $(y-objs-y)
list-multi		:= perfctr.o
endif

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

ifneq ($(VERSION).$(PATCHLEVEL),2.5)
perfctr.o:	$(perfctr-objs)
	$(LD) -r -o $@ $(perfctr-objs)
endif

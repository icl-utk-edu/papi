# $Id$
# Makefile for the Performance-monitoring counters driver.
#

SUB_DIRS :=
MOD_SUB_DIRS := $(SUB_DIRS)
ALL_SUB_DIRS := $(SUB_DIRS)

driver-objs-y := init.o
tests-objs-y :=
stub-objs-y :=
o-objs-y :=
ox-objs-m :=
mi-objs-m :=
m-objs-m :=

driver-objs-$(CONFIG_X86) += x86.o
tests-objs-$(CONFIG_X86) := x86_tests.o
kernel-objs-$(CONFIG_X86) := x86_setup.o

driver-objs-$(CONFIG_PERFCTR_INIT_TESTS) += $(tests-objs-y)

driver-objs-$(CONFIG_PERFCTR_VIRTUAL) += virtual.o
stub-objs-$(CONFIG_PERFCTR_VIRTUAL) += virtual_stub.o

driver-objs-$(CONFIG_PERFCTR_GLOBAL) += global.o

o-objs-$(CONFIG_PERFCTR) := $(kernel-objs-y) $(driver-objs-y)
ox-objs-$(CONFIG_PERFCTR) := $(kernel-objs-y) $(stub-objs-y)
mi-objs-$(CONFIG_PERFCTR) := $(driver-objs-y)
m-objs-$(CONFIG_PERFCTR) := perfctr.o

O_TARGET := kperfctr.o
O_OBJS := $(o-objs-y)
OX_OBJS := $(ox-objs-m)
MI_OBJS := $(mi-objs-m)
M_OBJS := $(m-objs-m)

include $(TOPDIR)/Rules.make

perfctr.o:	$(MI_OBJS)
	$(LD) $(LD_RFLAG) -r -o $@ $(MI_OBJS)

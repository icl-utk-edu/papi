# $Id: Makefile,v 1.20.2.4 2009/01/23 17:25:39 mikpe Exp $
# Makefile for the Performance-monitoring counters driver.

# We need -fno-unit-at-a-time with gcc-3.4 on x86 to avoid stack overflow.
# Kernels >= 2.6.6 do that automatically but older ones do not, so we
# unconditionally add that option here just in case.
my_check_gcc = $(shell if $(CC) $(CFLAGS) $(1) -S -o /dev/null -xc /dev/null > /dev/null 2>&1; then echo "$(1)"; else echo "$(2)"; fi ;)
EXTRA_CFLAGS_$(CONFIG_X86) := $(call my_check_gcc,-fno-unit-at-a-time,)
EXTRA_CFLAGS_$(CONFIG_X86_64) :=
EXTRA_CFLAGS_$(CONFIG_PPC32) :=
EXTRA_CFLAGS := $(EXTRA_CFLAGS_y)

# construct various object file lists:
# kernel-objs-y		kernel objects
# m-objs-m		perfctr.o if driver is module, empty otherwise
# driver-objs-y		objects for perfctr.o module, or empty

# This also covers x86_64.
driver-objs-$(CONFIG_X86) := x86.o
tests-objs-$(CONFIG_X86) := x86_tests.o
kernel-objs-$(CONFIG_X86) := x86_setup.o

driver-objs-$(CONFIG_PPC32) := ppc.o
tests-objs-$(CONFIG_PPC32) := ppc_tests.o
kernel-objs-$(CONFIG_PPC32) := ppc_setup.o

driver-objs-$(CONFIG_ARM) := arm.o
kernel-objs-$(CONFIG_ARM) := arm_setup.o

driver-objs-y += init.o marshal.o
driver-objs-$(CONFIG_PERFCTR_INIT_TESTS) += $(tests-objs-y)
driver-objs-$(CONFIG_PERFCTR_VIRTUAL) += virtual.o
stub-objs-$(CONFIG_PERFCTR)-$(CONFIG_PERFCTR_VIRTUAL) := virtual_stub.o
driver-objs-$(CONFIG_PERFCTR_GLOBAL) += global.o
m-objs-$(CONFIG_PERFCTR) := perfctr.o
kernel-objs-$(CONFIG_PERFCTR) += $(driver-objs-y)
kernel-objs-y += $(stub-objs-m-y)

perfctr-objs		:= $(driver-objs-y)
obj-m			+= $(m-objs-m)

obj-$(CONFIG_KPERFCTR)	+= kperfctr.o
kperfctr-objs		:= $(kernel-objs-y)

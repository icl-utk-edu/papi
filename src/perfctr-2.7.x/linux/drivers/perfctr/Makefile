# $Id: Makefile,v 1.28 2007/10/06 13:02:07 mikpe Exp $
# Makefile for the Performance-monitoring counters driver.

my_check_gcc = $(shell if $(CC) $(CFLAGS) $(1) -S -o /dev/null -xc /dev/null > /dev/null 2>&1; then echo "$(1)"; else echo "$(2)"; fi ;)
EXTRA_CFLAGS_$(CONFIG_X86) := $(call my_check_gcc,-fno-unit-at-a-time,)
EXTRA_CFLAGS_$(CONFIG_X86_64) :=
EXTRA_CFLAGS_$(CONFIG_PPC32) :=
EXTRA_CFLAGS_$(CONFIG_PPC64) :=
EXTRA_CFLAGS := $(EXTRA_CFLAGS_y)

# This also covers x86_64.
perfctr-objs-$(CONFIG_X86) := x86.o
tests-objs-$(CONFIG_X86) := x86_tests.o

perfctr-objs-$(CONFIG_PPC32) := ppc.o
tests-objs-$(CONFIG_PPC32) := ppc_tests.o

perfctr-objs-$(CONFIG_PPC64) := ppc64.o
tests-objs-$(CONFIG_PPC64) := ppc64_tests.o

perfctr-objs-y += init.o
perfctr-objs-$(CONFIG_PERFCTR_INIT_TESTS) += $(tests-objs-y)
perfctr-objs-$(CONFIG_PERFCTR_VIRTUAL) += virtual.o

perfctr-objs		:= $(perfctr-objs-y)
obj-$(CONFIG_PERFCTR)	:= perfctr.o

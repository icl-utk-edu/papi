# $Id$
# Makefile for the Performance-monitoring counters driver.

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

# $Id$
# Makefile for the Performance-monitoring counters driver.

ifeq ($(VERSION)$(PATCHLEVEL),24)
include Makefile24
else

# construct various object file lists:
# kernel-objs-y		kernel objects
# m-objs-m		perfctr.o if driver is module, empty otherwise
# driver-objs-y		objects for perfctr.o module, or empty

driver-objs-$(CONFIG_X86) := x86.o
tests-objs-$(CONFIG_X86) := x86_tests.o
kernel-objs-$(CONFIG_X86) := x86_setup.o

# Braindamage alert! x86_64 defines _both_ CONFIG_X86_64 and CONFIG_X86.
# These assignments need to override those above.
driver-objs-$(CONFIG_X86_64) := x86_64.o
tests-objs-$(CONFIG_X86_64) := x86_64_tests.o
kernel-objs-$(CONFIG_X86_64) := x86_64_setup.o

driver-objs-$(CONFIG_PPC32) := ppc.o
tests-objs-$(CONFIG_PPC32) := ppc_tests.o
kernel-objs-$(CONFIG_PPC32) := ppc_setup.o

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

endif	# ifeq 24

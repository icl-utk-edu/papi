KERNINC	= /usr/src/linux-2.4/include
PERFCTR ?= ./perfctr-2.6.x
PERFCTR_LIB_PATH = $(PERFCTR)/usr.lib
OPTFLAGS= -O3 -g -Wall
TOPTFLAGS= -g -Wall -DNO_SIMULT_EVENTSETS -D__x86_64__
#
# GNU G77 section
#
F77   = g77
FFLAGS        = -Dlinux
FOPTFLAGS= $(OPTFLAGS)
FTOPTFLAGS= $(TOPTFLAGS)
# #
# #  Portland Group PGF77 section
# #
# F77       = pgf77
# FFLAGS    = -Dlinux
# FOPTFLAGS = -O3 -tp p6
# FTOPTFLAGS= $(FOPTFLAGS)
# #
# #  Intel Corp. Fortran compiler
# #
# F77      = ifc
# FFLAGS   = -Dlinux
# LDFLAGS  = -lPEPCF90 -lIEPCF90 -lF90 -lintrins # Intel portability library (getarg_)
# FOPTFLAGS= -O3 -tpp6
# FTOPTFLAGS= $(FOPTFLAGS)

#
# DO NOT TOUCH BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING
#

LIBRARY = libpapi.a
SHLIB   = libpapi.so
SUBSTR	= perfctr-p3
MSUBSTR	= linux-perfctr-p3
MEMSUBSTR= linux
DESCR	= "Linux with PerfCtr 2.6.x patch for all Pentium IIIs"
LIBS	= static shared
TARGETS = serial multiplex_and_pthreads 

CC	= gcc
CC_SHR  = $(CC) -shared -fPIC -Xlinker "-soname" -Xlinker "libpapi.so" -Xlinker "-rpath" -Xlinker "$(DESTDIR)/lib"
CC_R	= $(CC) -pthread
CFLAGS  = -I$(PERFCTR)/usr.lib -I$(PERFCTR)/linux/include -I$(KERNINC) -I. -DPERFCTR20 -DPERFCTR24 -DPERFCTR25 -DSUBSTRATE=\"$(SUBSTR).h\" -DDEBUG
#-DDEBUG -DMPX_DEBUG -DMPX_DEBUG_TIMER
MISCSRCS= linux.c p3_events.c
MISCOBJS= linux.o p3_events.o marshal.o global.o misc.o virtual.o event_set.o event_set_amd.o
MISCHDRS= perfctr-p3.h
SHLIBDEPS = -L$(PERFCTR_LIB_PATH) -lperfctr

include Makefile.inc

linux.o: linux.c
	$(CC) $(CFLAGS) -c linux.c -o $@

p3_events.o: p3_events.c
	$(CC) $(CFLAGS) -c p3_events.c -o $@

marshal.o global.o misc.o virtual.o event_set.o event_set_amd.o: $(PERFCTR)/usr.lib/libperfctr.a
	ar x $(PERFCTR)/usr.lib/libperfctr.a

$(PERFCTR)/usr.lib/libperfctr.a:
	$(MAKE) -C $(PERFCTR)

native_clean:
	$(MAKE) -C $(PERFCTR) clean

native_install:
	-cp -p $(PERFCTR)/usr.lib/libperfctr.so $(DESTDIR)/lib
	-cp -p $(PERFCTR)/usr.lib/perfctr_event_codes.h $(DESTDIR)/include
	-cp -p $(PERFCTR)/usr.lib/libperfctr.h  $(DESTDIR)/include

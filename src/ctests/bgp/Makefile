BGP_SYSDIR         = /bgsys/drivers/ppcfloor
BGP_INSTALLDIR     = ${BGP_SYSDIR}
# BGP_INSTALLDIR     = /bgusr/dlherms/work
BGP_GNU_LINUX_PATH = ${BGP_SYSDIR}/gnu-linux
BGP_ARCH_PATH      = ${BGP_INSTALLDIR}/arch
BGP_RUNTIME_PATH   = ${BGP_INSTALLDIR}/runtime
BGP_PAPI_PATH      = /bgusr/mmcsfvt0/PAPIpatch
PREFIX             = BGP_PAPI_PATH
DESTDIR            =

LDFLAGS            = -L$(BGP_INSTALLDIR)/runtime/SPI -lSPI.cna
LDFLAGS_PAPI       = $(LDFLAGS) -L$(BGP_PAPI_PATH)/lib -lpapi

CC                 = $(SILENT) ${BGP_GNU_LINUX_PATH}/bin/powerpc-bgp-linux-gcc
CFLAGS             = -D_BGP -g -gdwarf-2 -O2 -Wall -I$(BGP_ARCH_PATH)/include -I$(BGP_PAPI_PATH)/include

TESTS   	= papi_1

bgp_tests: $(TESTS)

papi%:
	$(SILENT) $(CC) $(CFLAGS) -o $@ $@.c $(LDFLAGS_PAPI)

clean:
	rm -f core $(TESTS)  *~ *.o

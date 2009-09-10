CC=gcc
CFLAGS= #-DPAPIJ_DEBUG
PAPI_INCLUDE=..
JAVA_HOME=/pkgs/jdk1.5.0_02

PAPIJ_JAVA_SRC= EventSet.java            PAPI_granularity_option.java  PAPI_mh_tlb_info.java PAPI_mh_itlb_info.java \
                FlipInfo.java            PAPI_hw_info.java             PAPI_mh_level_info.java \
                FlopInfo.java            PAPI_inherit_option.java      PAPI_mh_utlb_info.java\
                IpcInfo.java             PapiJ.java                    PAPI_multiplex_option.java \
                PAPI_address_map.java    PAPI_mh_cache_info.java       PAPI_option.java \
                PAPI_debug_option.java   PAPI_mh_dcache_info.java      PAPI_overflow_option.java \
                PAPI_domain_option.java  PAPI_mh_dtlb_info.java        PAPI_preload_info.java \
                PAPI_event_info.java     PAPI_mh_icache_info.java      PAPI_mh_ucache_info.java PAPI_shlib_info.java \
                PAPI_exe_info.java       PAPI_mh_info.java             PAPI_substrate_info.java 

libpapij.so: Papi.o
	$(CC) -shared Papi.o -o $@

Papi.o: Papi.c PapiJ.h
	$(CC) $(CFLAGS) -fPIC -c -I$(PAPI_INCLUDE) -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/$(OSTYPE) $< -o $@

PapiJ.h: PapiJ.class
	$(JAVA_HOME)/bin/javah PapiJ

PapiJ.class: PapiJ.java PAPI_exe_info.java
	$(JAVA_HOME)/bin/javac $(PAPIJ_JAVA_SRC)

clean:
	/bin/rm -f libpapij.so Papi.o *.class PapiJ.h

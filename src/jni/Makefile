CC=gcc

PAPI_INCLUDE=..

PAPIJ_JAVA_SRC=EventSet.java PAPI_hw_info.java \
  FlopInfo.java PAPI_inherit_option.java \
  PAPI_debug_option.java PAPI_option.java \
  PAPI_domain_option.java PAPI_preload_option.java \
  PAPI_exe_info.java PAPI_preset_info.java \
  PAPI_granularity_option.java PapiJ.java

libpapij.so: Papi.o
	$(CC) -shared -o $@ $<

Papi.o: Papi.c PapiJ.h
	$(CC) $(CFLAGS) -fPIC -c -I$(PAPI_INCLUDE) -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/$(OSTYPE) $< -o $@

PapiJ.h: PapiJ.class
	javah PapiJ

PapiJ.class: PapiJ.java PAPI_exe_info.java
	javac $(PAPIJ_JAVA_SRC)

clean:
	/bin/rm -f libpapij.so Papi.o *.class PapiJ.h

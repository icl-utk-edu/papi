.PHONY: java_package

CC=gcc
CFLAGS= -DPAPIJ_DEBUG
PAPI_INCLUDE=..
# JAVA_HOME=/pkgs/jdk1.5.0_02
JAVA_HOME=/usr/lib/jvm/java-6-sun-1.6.0.06
JAVAH=$(JAVA_HOME)/bin/javah
JAVAC=$(JAVA_HOME)/bin/javac
JNI_INC=-I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/$(OSTYPE)

libpapij.so: Papi.o
	$(CC) -shared Papi.o -o $@

Papi.o: Papi.c PapiJ.h
	$(CC) $(CFLAGS) -fPIC -c -I$(PAPI_INCLUDE) $(JNI_INC) $< -o $@

PapiJ.h: java_package
	$(JAVAH) edu.utk.cs.icl.papi.PapiJ

java_package: edu/utk/cs/icl/papi/*.java
	$(JAVAC) edu/utk/cs/icl/papi/*.java

clean:
	/bin/rm -f libpapij.so Papi.o edu/utk/cs/icl/papi/*.class edu_utk_cs_icl_papi_PapiJ.h

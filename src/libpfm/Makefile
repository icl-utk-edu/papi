#
# Copyright (C) 2001 Hewlett-Packard Co
# Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com>
#
TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/..

include ./config.mk
include ./rules.mk

EVENT_DB=perfmon.database

SRCS=pfmlib.c pfmusr.c pfmerror.c
OBJS=pfmlib.o pfmusr.o pfmerror.o

INCDIR=-I$(PFM_INC_DIR) -I.

TARGET=libpfm.a


all: $(TARGET)

$(TARGET):  pme_list.h $(OBJS)
	$(RM) $@
	$(AR) qv $@ $(OBJS)

pme_list.h: $(EVENT_DB) genpme.py
	./genpme.py $(EVENT_DB) >$@

clean:
	rm -f $(OBJS) $(TARGET)

distclean: clean
	rm -f pme_list.h

depend:
	$(MKDEP) $(SRCS)

$(OBJS) : pme_list.h

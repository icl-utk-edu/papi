#
# Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
# Contributed by Stephane Eranian <eranian@hpl.hp.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.  
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/../..

include $(TOPDIR)/config.mk
include $(TOPDIR)/rules.mk

LIBS += -lm

ifeq ($(SYS),Linux)
TARGETS=smpl_p4_pebs smpl_core_pebs smpl_amd64_ibs smpl_nhm_lbr smpl_pebs
endif

all: $(TARGETS)

$(TARGETS):  %:%.o $(PFMLIB) ../detect_pmcs.o
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(LIBS)

clean:
	$(RM) -f *.o $(TARGETS) *~

distclean: clean

install_examples: $(TARGETS)

install_examples:
	@echo installing: $(TARGETS)
	-mkdir -p $(DESTDIR)$(EXAMPLESDIR)/v2/x86
	$(INSTALL) -m 755 $(TARGETS) $(DESTDIR)$(EXAMPLESDIR)/v2/x86

#
# examples are installed as part of the RPM install, typically in /usr/share/doc/libpfm-X.Y/
#

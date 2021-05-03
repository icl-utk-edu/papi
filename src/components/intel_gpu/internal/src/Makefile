#This makefile for generating related binaries using level0 API

CXX_INCLUDE+=$(INCLUDE)
CXX_INCLUDE+=-I/usr/include -I../inc
INTEL_L0_HEADERS ?=/usr/include
CFLAGS=
CXX_FLAGS=-std=c++11

OBJDIR=../lib64

TARGETS=$(OBJDIR)/GPUMetricInterface.o $(OBJDIR)/GPUMetricHandler.o

all:   $(TARGETS)


$(OBJDIR)/GPUMetricInterface.o: GPUMetricInterface.cpp  ../inc/GPUMetricInterface.h
	$(CXX) -g -fpic $(CPPFLAGS) $(CXX_INCLUDE) -I$(INTEL_L0_HEADERS) -c GPUMetricInterface.cpp -o $(OBJDIR)/GPUMetricInterface.o

$(OBJDIR)/GPUMetricHandler.o: GPUMetricHandler.cpp ../inc/GPUMetricInterface.h ../inc/GPUMetricHandler.h
	$(CXX) -g -fpic $(CPPFLAGS) $(CXX_INCLUDE) -I$(INTEL_L0_HEADERS)  -c GPUMetricHandler.cpp -o $(OBJDIR)/GPUMetricHandler.o


clean:
	rm -f *.o  $(OBJDIR)/*.o
	

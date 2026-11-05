CC ?= gcc
SDE_INC = -I. -I..
SDE_LD = -ldl -pthread
CFLAGS += -Wextra -Wall -O2

SONAME = libsde.so.1
REALNAME = libsde.so.1.0

%_d.o: %.c
		$(CC) -c -Bdynamic -fPIC -shared -fvisibility=hidden $(CFLAGS) $(SDE_INC) $< -o $@
%_s.o: %.c
		$(CC) -c -Bstatic -static $(CFLAGS) $(SDE_INC) $< -o $@

DOBJS=$(patsubst %.c,%_d.o,$(wildcard *.c))
SOBJS=$(patsubst %.c,%_s.o,$(wildcard *.c))

all: dynamic static

dynamic: $(DOBJS)
	$(CC) $(LDFLAGS) -Bdynamic -fPIC -shared -Wl,-soname -Wl,$(SONAME) -fvisibility=hidden $(CFLAGS) $(DOBJS) -lrt -ldl -pthread -o $(REALNAME)
	rm -f *_d.o

static: $(SOBJS)
	ar rs libsde.a $(SOBJS)
	rm -f *_s.o

clean:
	rm -f *.o $(REALNAME) libsde.a


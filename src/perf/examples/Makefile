CFLAGS=-Wall -g -I..
LDFLAGS=-L.. -lperf

BINS=psys pfork pwait pflops pfast psumchild pconfig
all: $(BINS)
clean:
	rm -f $(BINS) *~
psys: psys.c
	gcc $(CFLAGS) psys.c -o psys $(LDFLAGS)
pfork: pfork.c
	gcc $(CFLAGS) pfork.c -o pfork $(LDFLAGS)
pwait: pwait.c
	gcc $(CFLAGS) pwait.c -o pwait $(LDFLAGS)
pflops: pflops.c
	gcc $(CFLAGS) pflops.c -o pflops $(LDFLAGS)
pfast: pfast.c
	gcc $(CFLAGS) pfast.c -o pfast $(LDFLAGS)
psumchild: psumchild.c
	gcc $(CFLAGS) psumchild.c -o psumchild $(LDFLAGS)
pconfig: pconfig.c
	gcc $(CFLAGS) pconfig.c -o pconfig $(LDFLAGS)

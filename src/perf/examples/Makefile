CFLAGS=-Wall -g -I..
LDFLAGS=-L.. -lperf

BINS=pmesi pcycles psys pfork pwait pflops pfast psumchild pconfig pfastwrite
all: $(BINS)
clean:
	rm -f $(BINS) *~
pmesi: pmesi.c
	gcc $(CFLAGS) pmesi.c -o pmesi $(LDFLAGS)
pcycles: pcycles.c
	gcc $(CFLAGS) pcycles.c -o pcycles $(LDFLAGS)
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
pfastwrite: pfastwrite.c
	gcc $(CFLAGS) pfastwrite.c -o pfastwrite $(LDFLAGS)

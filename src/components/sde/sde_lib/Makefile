SDE_INC = -I.
SDE_LD = -ldl -pthread

F90FLAGS += -ffree-form -ffree-line-length-none -fPIC

all: obj
		make link_em

obj: sde_lib.c sde_common.c weak_symbols.c
		-$(CC) -c -Bdynamic -fPIC -shared -Wextra -Wall -O2 $(SDE_INC) weak_symbols.c -o weak_symbols_d.o
		-$(CC) -c -Bdynamic -fPIC -shared -D_GNU_SOURCE -fvisibility=hidden -Wextra -Wall -O2 $(SDE_INC) sde_lib.c -o sde_lib_d.o
		-$(CC) -c -Bdynamic -fPIC -shared -D_GNU_SOURCE -fvisibility=hidden -Wextra -Wall -O2 $(SDE_INC) sde_common.c -o sde_common_d.o
		-$(CC) -c -Bstatic -static -Wextra -Wall -O2 $(SDE_INC) weak_symbols.c -o weak_symbols_s.o
		-$(CC) -c -Bstatic -static -DSTATIC_SDE -D_GNU_SOURCE -Wextra -Wall -O2 $(SDE_INC) sde_lib.c -o sde_lib_s.o
		-$(CC) -c -Bstatic -static -D_GNU_SOURCE -Wextra -Wall -O2 $(SDE_INC) sde_common.c -o sde_common_s.o

link_em:
		$(CC) -Bdynamic -fPIC -shared -Wl,-soname -Wl,libsde.so -D_GNU_SOURCE -fvisibility=hidden -Wextra -Wall -g -O2 $(wildcard *_d.o) -lrt -ldl -pthread -o libsde.so
		ar rs libsde.a $(wildcard *_s.o)

clean:
	rm -f *.o libsde.so libsde.a

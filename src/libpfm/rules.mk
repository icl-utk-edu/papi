#
# Copyright (C) 2001 Hewlett-Packard Co
# Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com>
#
#
.SUFFIXES: .c .S .o

.S.o:
	$(CC) $(INCDIR) $(CFLAGS) -c $*.S
.c.o:
	$(CC) $(INCDIR) $(CFLAGS) -c $*.c



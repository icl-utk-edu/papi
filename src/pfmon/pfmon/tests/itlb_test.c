/*
 * Copyright (C) 2002 Hewlett-Packard Co
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file is part of pfmon, a sample tool to measure performance 
 * of applications on Linux/ia64.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
 */

#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

typedef struct {
	unsigned long addr;
	unsigned long gp;
} func_desc_t;

#define FUNC(n) int func_##n(void) { return n; }

FUNC(1) FUNC(2) FUNC(3) FUNC(4) FUNC(5) FUNC(6) FUNC(7) FUNC(8) FUNC(9)
FUNC(10) FUNC(11) FUNC(12) FUNC(13) FUNC(14) FUNC(15) FUNC(16) FUNC(17) FUNC(18) FUNC(19)
FUNC(20) FUNC(21) FUNC(22) FUNC(23) FUNC(24) FUNC(25) FUNC(26) FUNC(27) FUNC(28) FUNC(29)
FUNC(30) FUNC(31) FUNC(32) FUNC(33) FUNC(34) FUNC(35) FUNC(36) FUNC(37) FUNC(38) FUNC(39)
FUNC(40) FUNC(41) FUNC(42) FUNC(43) FUNC(44) FUNC(45) FUNC(46) FUNC(47) FUNC(48) FUNC(49)
FUNC(50) FUNC(51) FUNC(52) FUNC(53) FUNC(54) FUNC(55) FUNC(56) FUNC(57) FUNC(58) FUNC(59)
FUNC(60) FUNC(61) FUNC(62) FUNC(63) FUNC(64)

static int (*tab[])(void)={
	func_1, func_2, func_3, func_4, func_5, func_6, func_7, func_8, func_9,
	func_10, func_11, func_12, func_13, func_14, func_15, func_16, func_17, func_18, func_19,
	func_20, func_21, func_22, func_23, func_24, func_25, func_26, func_27, func_28, func_29,
	func_30, func_31, func_32, func_33, func_34, func_35, func_36, func_37, func_38, func_39,
	func_40, func_41, func_42, func_43, func_44, func_45, func_46, func_47, func_48, func_49,
	func_50, func_51, func_52, func_53, func_54, func_55, func_56, func_57, func_58, func_59,
	func_60, func_61, func_62, func_63, func_64,
	NULL
};

int 
doit(unsigned long iter, unsigned int max)
{
	unsigned int sum = 0, i, j;
	int (**pf)(void);

	for(j=0; j < iter; j++) {
		for(i=0, pf = tab; i < max && *pf; i++, pf++) {
			sum += (**pf)();
		}
	}
	return sum; /* ensures the compiler does not get rid of everything */
}


int 
main(int argc, char **argv)
{
	func_desc_t *fd1, *fd2;
	int pgsz;
	unsigned long iter;
	unsigned int max = -1;

	pgsz = getpagesize();

	fd1 = (func_desc_t *)func_1;
	fd2 = (func_desc_t *)func_2;

	if ((fd2->addr-fd1->addr) != pgsz) {
		printf("the program was not compiled with -falign-funtions=%d\n", pgsz);
		exit(1);
	}

	iter = argc > 1 ?strtoul(argv[1], NULL, 10) : 10000; 
	max  = argc > 2 ? atoi(argv[2]) : -1;

	doit(iter, max);
	_exit(0); /* short circuit libc exit() */
}

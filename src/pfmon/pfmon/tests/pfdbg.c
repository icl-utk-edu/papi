/*
 * pfdbg.c: toggle the debug behavior of the kernel perfmon system
 *
 * Copyright (C) 2001 Hewlett-Packard Co
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
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>

#include <perfmon/perfmon.h>

#define PFDBG_VERSION	"0.03"

static struct option cmd_options[]={
	{ "version", 0, 0, 0},
	{ "on", 0, 0, 1},
	{ "off", 0, 0, 2},
	{ "help", 0, 0, 3},
	{ 0, 0, 0, 0}
};

static void fatal_error(char *fmt,...) __attribute__((noreturn));

static void
fatal_error(char *fmt, ...) 
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	exit(1);
}


void
usage(char **argv)
{
	printf("Usage: %s [OPTIONS]... COMMAND\n", argv[0]);

	printf(	"-h, --help\tdisplay this help and exit\n"
		"--version\toutput version information and exit\n"
		"--on\t\tturn on debugging\n"
		"--off\t\tturn off debugging\n"
	);
}

void
debug_on_off(unsigned int mode)
{
	if ( perfmonctl(0, PFM_DEBUG, &mode, 1) < 0) {
		if (errno == ENOSYS) 
			fatal_error("No perfmon support in kernel\n");
		else
			fatal_error("perfmon does not support DEBUG option\n");
	}
}

int
main(int argc, char **argv)
{
	int c;

	while ((c=getopt_long(argc, argv,"h", cmd_options, 0)) != -1) {
		switch(c) {
			case   0:
				printf("Version %s Date: %s\n", PFDBG_VERSION, __DATE__);
				exit(0);
			case   1:
				debug_on_off(1U);
				break;
			case   2:
				debug_on_off(0U);
				break;
			case   3:
			case 'h':
				usage(argv);
				break;
			default:
				fatal_error("Unknown option\n");
		}
	}
	return 0;
}

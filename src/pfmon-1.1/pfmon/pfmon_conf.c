/*
 * pfmon_conf.c  - support for pfmon configuration file
 *
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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#include <ctype.h>
#include <pwd.h>
#include <sys/param.h>

#include "pfmon.h"

#define CONFIG_DIR		EXEC_PREFIX"/lib/pfmon"
#define CONFIG_FILE		"pfmon.conf"

#define MAX_OPTION_NAMELEN	32
#define MAX_OPTION_VALUELEN	32


typedef struct _config_option {
	struct _config_option *next;

	char	name[MAX_OPTION_NAMELEN];
	union {
		char		val_str[MAX_OPTION_VALUELEN];
		unsigned long	val_long;
	} value;
} config_option_t;

#define val_long	value.val_long
#define val_str		value.val_str

typedef struct _config_section {
		struct _conf_section	*next;		/* pointer to next section */
		int			sec_idx;	/* index of section is section_desc */
		int			unused;
		config_option_t		*options;	/* all defined options for this section */
} config_section_t;

static config_option_t *matcher;

int
find_opcode_matcher(char *name, unsigned long *val)
{
	config_option_t *p;

	for(p = matcher; p ; p = p->next) {
		if (!strcmp(name, p->name)) {
			if (val) *val = p->val_long;
			return 1;
		}
	}
	return 0;
}

static void
add_matcher(char *name, unsigned long val)
{
	config_option_t *n;

	if (find_opcode_matcher(name, NULL)) fatal_error("opcode matcher %s defined twice\n", name);

	if (isdigit(name[0])) fatal_error("invalid opcode matcher name %s: cannot begin with digit\n", name);

	n = (config_option_t *)malloc(sizeof(config_option_t));
	if (n == NULL) fatal_error("cannot allocate new config option\n");

	strcpy(n->name, name);
	n->val_long = val;
	n->next = matcher;

	matcher = n;
}

void
print_opcode_matchers(void)
{
	config_option_t *p;

	for(p = matcher; p ; p = p->next) {
		printf("%s=0x%016lx\n", p->name, p->val_long);
	}
}

static void
parse_config_file(FILE *fp, char *filename)
{
	char name[MAX_OPTION_NAMELEN];
	unsigned long val;

	while(fscanf(fp,"%s 0x%lx", name, &val) == 2) {
		add_matcher(name, val);
	}
}

void
load_config_file(void)
{
	FILE *fp;
	struct passwd *pwd;
	char filename[PATH_MAX];

	pwd = getpwuid(getuid());
	if (pwd == NULL) fatal_error("Illegal user id %d\n", getuid());

	sprintf(filename, "%s/.%s", pwd->pw_dir, CONFIG_FILE);

	fp = fopen(filename, "r");
	if (fp) goto found;

	sprintf(filename, "%s/%s", CONFIG_DIR, CONFIG_FILE);


	fp = fopen(filename, "r");
	if (fp == NULL) return; /* no config file */
found:
	vbprintf("using config file: %s\n", filename);

	parse_config_file(fp, filename);

	fclose(fp);
}

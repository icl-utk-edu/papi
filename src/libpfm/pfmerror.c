/*
 * pfmlib.c
 *
 * Copyright (C) 2001 Hewlett-Packard Co
 * Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com>
 */

#include <sys/types.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include "pfmlib.h"

#include "pfm_private.h"

static const char *pfm_err_lst[]={
	"Unknown event",
	"Invalid threshold"
};

static const int pfm_err_count = sizeof(pfm_err_lst)/sizeof(char *);

const char *
pfm_strerror(int error)
{
	error -= PFME_BASE;
	if (error < 0 || error >= pfm_err_count) return (const char *)"??";

	return pfm_err_lst[error];
}

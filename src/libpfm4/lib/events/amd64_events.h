/*
 * Copyright (c) 2006, 2007 Advanced Micro Devices, Inc.
 * Contributed by Ray Bryant <raybry@mpdtxmail.amd.com> 
 * Contributed by Robert Richter <robert.richter@amd.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * This file is part of libpfm, a performance monitoring support library for
 * applications on Linux.
 */

#include "amd64_events_k7.h"
#include "amd64_events_k8.h"
#include "amd64_events_fam10h.h"

struct amd64_table {
	int			num;
	const amd64_entry_t	*events;
};

static const struct amd64_table amd64_k7_table = {
	.num	  = PME_AMD64_K7_EVENT_COUNT,
	.events	  = amd64_k7_pe,
};

static const struct amd64_table amd64_k8_table = {
	.num	  = PME_AMD64_K8_EVENT_COUNT,
	.events	  = amd64_k8_pe,
};

static const struct amd64_table amd64_fam10h_table = {
	.num      = PME_AMD64_FAM10H_EVENT_COUNT,
	.events   = amd64_fam10h_pe,
};

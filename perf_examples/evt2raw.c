/*
 * evt2raw.c - example which converts an event string (event + modifiers) to
 * a raw event code usable by the perf tool.
 *
 * Copyright (c) 2010 IBM Corp.
 * Contributed by Corey Ashford <cjashfor@us.ibm.com>
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
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <err.h>
#include <perfmon/pfmlib_perf_event.h>

static void
usage(void)
{
	printf("usage: evt2raw <event>\n"
		"<event> is the symbolic event, including modifiers, to "
		"translate to a raw code.\n");
}

int
main(int argc, char **argv)
{
	int ret;
	struct perf_event_attr pea;
	char *event_str;

	if (argc != 2) {
		usage();
		return 1;
	}
	event_str = argv[1];

	ret = pfm_initialize();
	if (ret != PFM_SUCCESS)
		errx(1, "Internal error: pfm_initialize returned %s\n",
			pfm_strerror(ret));

	ret = pfm_get_perf_event_encoding(event_str, PFM_PLM0|PFM_PLM3, &pea,
		NULL, NULL);
	if (ret != PFM_SUCCESS)
		errx(1, "Error: pfm_get_perf_encoding returned %s\n",
			pfm_strerror(ret));

	if (pea.type != PERF_TYPE_RAW)
		errx(1, "Error: %s is not a raw hardware event\n", event_str);

	printf("r%"PRIx64"\n", pea.config);

	return 0;
}

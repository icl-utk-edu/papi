/*
 * perf_util.c - helper functions for perf_events
 *
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmail.com>
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
#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>

#include <perfmon/pfmlib_perf_event.h>
#include "perf_util.h"

/* the **fd parameter must point to a null pointer on the first call
 * max_fds and num_fds must both point to a zero value on the first call
 * The return value is success (0) vs. failure (non-zero)
 */
int
perf_setup_argv_events(const char **argv, perf_event_desc_t **fds, int *num_fds)
{
	perf_event_desc_t *fd;
	int new_max, ret, num, max_fds;
	int group_leader;

	if (!(argv && fds && num_fds))
		return -1;

	fd = *fds;
	if (fd) {
		max_fds = fd[0].max_fds;
		if (max_fds < 2)
			return -1;
		num = *num_fds;
	} else {
		max_fds = num = 0; /* bootstrap */
	}
	group_leader = num;

	while(*argv) {
		if (num == max_fds) {
			if (max_fds == 0)
				new_max = 2;
			else
				new_max = max_fds << 1;

			if (new_max < max_fds) {
				warn("too many entries");
				goto error;
			}
			fd = realloc(fd, new_max * sizeof(*fd));
			if (!fd) {
				warn("cannot allocate memory");
				goto error;
			}
			/* reset newly allocated chunk */
			memset(fd + max_fds, 0, (new_max - max_fds) * sizeof(*fd));
			max_fds = new_max;

			/* update max size */
			fd[0].max_fds = max_fds;
		}

		ret = pfm_get_perf_event_encoding(*argv, PFM_PLM3, &fd[num].hw, NULL, NULL);
		if (ret != PFM_SUCCESS) {
			warnx("event %s: %s\n", *argv, pfm_strerror(ret));
			goto error;
		}
		/* ABI compatibility */
		fd[num].hw.size = sizeof(struct perf_event_attr);

		fd[num].name = *argv;
		fd[num].group_leader = group_leader;
		num++;
		argv++;
	}
	*num_fds = num;
	*fds = fd;
	return 0;
error:
	free(fd);
	return -1;
}

int
perf_setup_list_events(const char *ev, perf_event_desc_t **fd, int *num_fds)
{
	const char **argv;
	char *p, *q, *events;
	int i, ret, num = 0;

	if (!(ev && fd && num_fds))
		return -1;

	events = strdup(ev);
	if (!events)
		return -1;

	q = events;
	while((p = strchr(q, ','))) {
		num++;
		q = p + 1;
	}
	num++;
	num++; /* terminator */

	argv = malloc(num * sizeof(char *));
	if (!argv) {
		free(events);
		return -1;
	}

	for(i=0, q = events; i < num-2; i++, q = p + 1) {
		p = strchr(q, ',');
		*p = '\0';
		argv[i] = q;
	}
	argv[i++] = q;
	argv[i] = NULL;
	ret = perf_setup_argv_events(argv, fd, num_fds);
	free(argv);
	return ret;
}

int
perf_get_group_nevents(perf_event_desc_t *fds, int num, int idx)
{
	int leader;
	int i;

	if (idx < 0 || idx >= num)
		return 0;

	leader = fds[idx].group_leader;

	for (i = leader + 1; i < num; i++) {
		if (fds[i].group_leader != leader) {
			/* This is a new group leader, so the previous
			 * event was the final event of the preceding
			 * group.
			 */
			return i - leader;
		}
	}
	return i - leader;
}

int
perf_read_buffer(struct perf_event_mmap_page *hdr, size_t pgmsk, void *buf, size_t sz)
{
	char *data;
	unsigned long tail, head;
	size_t avail_sz, m, c;
	
	/*
 	 * data ipoint to start of buffer payload
 	 * first page is buffer header
 	 */
	data = (char *)(((unsigned long)hdr)+sysconf(_SC_PAGESIZE));

	/*
 	 * position of head and tail within the buffer payload
 	 */
	tail = hdr->data_tail & pgmsk;
	head = hdr->data_head & pgmsk;

	/*
 	 * size of what was added
 	 * data_head, data_tail never wrap around
 	 */
	avail_sz = hdr->data_head - hdr->data_tail;
	if (sz > avail_sz)
		return -1;

	/*
 	 * straddles if:
 	 *        head (modulo pgmsk) < tail + size
 	 * otherwise fits into he buffer
 	 */
	if ((tail + avail_sz) == head) {
		memcpy(buf, &data[tail], sz);
	} else {
		/*
		 * c = size till end of buffer
 		 */
		c = pgmsk + 1 -  tail;

		/*
		 * min with requested size
		 */
		m = c < sz ? c : sz;

		/* copy beginning */
		memcpy(buf, data+tail, m);

		if ((sz - m) > 0)
			memcpy(buf+m, &data[0], sz - m);
	}
	hdr->data_tail += sz;
	return 0;
}

void
perf_skip_buffer(struct perf_event_mmap_page *hdr, size_t sz)
{
	if ((hdr->data_tail + sz) > hdr->data_head)
		sz = hdr->data_head - hdr->data_tail;

	hdr->data_tail += sz;
}

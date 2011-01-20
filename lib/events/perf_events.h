/*
 * Copyright (c) 2009 Google, Inc
 * Contributed by Stephane Eranian <eranian@google.com>
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
static perf_event_t perf_static_events[]={
	PCL_EVT(PERF_COUNT_HW_CPU_CYCLES, PERF_TYPE_HARDWARE, PERF_ATTR_HW),
	PCL_EVT(PERF_COUNT_HW_INSTRUCTIONS, PERF_TYPE_HARDWARE, PERF_ATTR_HW),
	PCL_EVT(PERF_COUNT_HW_CACHE_REFERENCES, PERF_TYPE_HARDWARE, PERF_ATTR_HW),
	PCL_EVT(PERF_COUNT_HW_CACHE_MISSES, PERF_TYPE_HARDWARE, PERF_ATTR_HW),
	PCL_EVT(PERF_COUNT_HW_BRANCH_INSTRUCTIONS, PERF_TYPE_HARDWARE, PERF_ATTR_HW),
	PCL_EVT(PERF_COUNT_HW_BRANCH_MISSES, PERF_TYPE_HARDWARE, PERF_ATTR_HW),
	PCL_EVT(PERF_COUNT_HW_BUS_CYCLES, PERF_TYPE_HARDWARE, PERF_ATTR_HW),

        PCL_EVT(PERF_COUNT_SW_CPU_CLOCK, PERF_TYPE_SOFTWARE, PERF_ATTR_SW),
        PCL_EVT(PERF_COUNT_SW_TASK_CLOCK, PERF_TYPE_SOFTWARE, PERF_ATTR_SW),
        PCL_EVT(PERF_COUNT_SW_PAGE_FAULTS , PERF_TYPE_SOFTWARE, PERF_ATTR_SW),
        PCL_EVT(PERF_COUNT_SW_CONTEXT_SWITCHES, PERF_TYPE_SOFTWARE, PERF_ATTR_SW),
        PCL_EVT(PERF_COUNT_SW_CPU_MIGRATIONS, PERF_TYPE_SOFTWARE, PERF_ATTR_SW),
        PCL_EVT(PERF_COUNT_SW_PAGE_FAULTS_MIN, PERF_TYPE_SOFTWARE, PERF_ATTR_SW),
        PCL_EVT(PERF_COUNT_SW_PAGE_FAULTS_MAJ, PERF_TYPE_SOFTWARE, PERF_ATTR_SW),
	{
	.name = "PERF_COUNT_HW_CACHE_L1D",
	.desc = "L1 data cache",
	.id   = PERF_COUNT_HW_CACHE_L1D,
	.type = PERF_TYPE_HW_CACHE,
	.numasks = 5,
	.modmsk = PERF_ATTR_HW,
	.umask_ovfl_idx = -1,
	.ngrp = 2,
	.umasks = {
		{ .uname = "READ",
		  .udesc = "read access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_READ << 8,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 0,
		},
		{ .uname = "WRITE",
		  .udesc = "write access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_WRITE << 8,
		  .grpid = 0,
		},
		{ .uname = "PREFETCH",
		  .udesc = "prefetch access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_PREFETCH << 8,
		  .grpid = 0,
		},
		{ .uname = "ACCESS",
		  .udesc = "hit access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16,
		  .grpid = 1,
		},
		{ .uname = "MISS",
		  .udesc = "miss access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_MISS << 16,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 1,
		}
	}
       },
       {
	.name = "PERF_COUNT_HW_CACHE_L1I",
	.desc = "L1 instruction cache",
	.id   = PERF_COUNT_HW_CACHE_L1I,
	.type = PERF_TYPE_HW_CACHE,
	.numasks = 4,
	.modmsk = PERF_ATTR_HW,
	.umask_ovfl_idx = -1,
	.ngrp = 2,
	.umasks = {
		{ .uname = "READ",
		  .udesc = "read access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_READ << 8,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 0,
		},
		{ .uname = "PREFETCH",
		  .udesc = "prefetch access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_PREFETCH << 8,
		  .grpid = 0,
		},
		{ .uname = "ACCESS",
		  .udesc = "hit access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16,
		  .grpid = 1,
		},
		{ .uname = "MISS",
		  .udesc = "miss access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_MISS << 16,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 1,
		}
	}
       },
       {
	.name = "PERF_COUNT_HW_CACHE_LL",
	.desc = "Last level cache",
	.id   = PERF_COUNT_HW_CACHE_LL,
	.type = PERF_TYPE_HW_CACHE,
	.numasks = 5,
	.modmsk = PERF_ATTR_HW,
	.umask_ovfl_idx = -1,
	.ngrp = 2,
	.umasks = {
		{ .uname = "READ",
		  .udesc = "read access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_READ << 8,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 0,
		},
		{ .uname = "WRITE",
		  .udesc = "write access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_WRITE << 8,
		  .grpid = 0,
		},
		{ .uname = "PREFETCH",
		  .udesc = "prefetch access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_PREFETCH << 8,
		  .grpid = 0,
		},
		{ .uname = "ACCESS",
		  .udesc = "hit access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16,
		  .grpid = 1,
		},
		{ .uname = "MISS",
		  .udesc = "miss access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_MISS << 16,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 1,
		}
	}
       },
       {
	.name = "PERF_COUNT_HW_CACHE_DTLB",
	.desc = "Data Translation Lookaside Buffer",
	.id   = PERF_COUNT_HW_CACHE_DTLB,
	.type = PERF_TYPE_HW_CACHE,
	.numasks = 5,
	.modmsk = PERF_ATTR_HW,
	.umask_ovfl_idx = -1,
	.ngrp = 2,
	.umasks = {
		{ .uname = "READ",
		  .udesc = "read access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_READ << 8,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 0,
		},
		{ .uname = "WRITE",
		  .udesc = "write access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_WRITE << 8,
		  .grpid = 0,
		},
		{ .uname = "PREFETCH",
		  .udesc = "prefetch access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_PREFETCH << 8,
		  .grpid = 0,
		},
		{ .uname = "ACCESS",
		  .udesc = "hit access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16,
		  .grpid = 1,
		},
		{ .uname = "MISS",
		  .udesc = "miss access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_MISS << 16,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 1,
		}
	}
       },
       {
	.name = "PERF_COUNT_HW_CACHE_ITLB",
	.desc = "Instruction Translation Lookaside Buffer",
	.id   = PERF_COUNT_HW_CACHE_ITLB,
	.type = PERF_TYPE_HW_CACHE,
	.numasks = 3,
	.modmsk = PERF_ATTR_HW,
	.umask_ovfl_idx = -1,
	.ngrp = 2,
	.umasks = {
		{ .uname = "READ",
		  .udesc = "read access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_READ << 8,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 0,
		},
		{ .uname = "ACCESS",
		  .udesc = "hit access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16,
		  .grpid = 1,
		},
		{ .uname = "MISS",
		  .udesc = "miss access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_MISS << 16,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 1,
		}
	}
       },
       {
	.name = "PERF_COUNT_HW_CACHE_BPU",
	.desc = "Branch Prediction Unit",
	.id   = PERF_COUNT_HW_CACHE_BPU,
	.type = PERF_TYPE_HW_CACHE,
	.numasks = 3,
	.modmsk = PERF_ATTR_HW,
	.umask_ovfl_idx = -1,
	.ngrp = 2,
	.umasks = {
		{ .uname = "READ",
		  .udesc = "read access",
		  .uid   = PERF_COUNT_HW_CACHE_OP_READ << 8,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 0,
		},
		{ .uname = "ACCESS",
		  .udesc = "hit access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16,
		  .grpid = 1,
		},
		{ .uname = "MISS",
		  .udesc = "miss access",
		  .uid   = PERF_COUNT_HW_CACHE_RESULT_MISS << 16,
		  .uflags= PERF_FL_DEFAULT,
		  .grpid = 1,
		}
	}
       }
};
#define PME_PERF_EVENT_COUNT (sizeof(perf_static_events)/sizeof(perf_event_t))

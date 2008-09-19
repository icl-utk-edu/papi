/*
 * Copyright (c) 2008 Google, Inc
 * Contributed by Stephane Eranian <eranian@gmai.com>
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

/* table 18.11 */
#define INTEL_ATOM_MESI \
		{ .pme_uname = "MESI",\
		  .pme_udesc = "Any cacheline access",\
		  .pme_ucode = 0xf\
		},\
		{ .pme_uname = "I_STATE",\
		  .pme_udesc = "Invalid cacheline",\
		  .pme_ucode = 0x1\
		},\
		{ .pme_uname = "S_STATE",\
		  .pme_udesc = "Shared cacheline",\
		  .pme_ucode = 0x2\
		},\
		{ .pme_uname = "E_STATE",\
		  .pme_udesc = "Exclusive cacheline",\
		  .pme_ucode = 0x4\
		},\
		{ .pme_uname = "M_STATE",\
		  .pme_udesc = "Modified cacheline",\
		  .pme_ucode = 0x8\
		}

/* table 18.9 */
#define INTEL_ATOM_AGENT \
		{ .pme_uname = "THIS_AGENT",\
		  .pme_udesc = "This agent",\
		  .pme_ucode = 0x00\
		},\
		{ .pme_uname = "ALL_AGENTS",\
		  .pme_udesc = "Any agent on the bus",\
		  .pme_ucode = 0x20\
		}

/* table 18.8 */
#define INTEL_ATOM_CORE \
		{ .pme_uname = "SELF",\
		  .pme_udesc = "This core",\
		  .pme_ucode = 0x40\
		},\
		{ .pme_uname = "BOTH_CORES",\
		  .pme_udesc = "Both cores",\
		  .pme_ucode = 0xc0\
		}

/* table 18.10 */
#define INTEL_ATOM_PREFETCH \
		{ .pme_uname = "ANY",\
		  .pme_udesc = "All inclusive",\
		  .pme_ucode = 0x30\
		},\
		{ .pme_uname = "PREFETCH",\
		  .pme_udesc = "Hardware prefetch only",\
		  .pme_ucode = 0x10\
		}

static pme_intel_atom_entry_t intel_atom_pe[]={
	/*
	 * BEGIN architectural perfmon events
	 */
/* 0 */{.pme_name  = "UNHALTED_CORE_CYCLES",
	.pme_code  = 0x003c,
	.pme_flags = PFMLIB_INTEL_ATOM_FIXED1,
	.pme_desc  = "Unhalted core cycles",
       },
/* 1 */{.pme_name  = "UNHALTED_REFERENCE_CYCLES",
	.pme_code  = 0x013c,
	.pme_flags = PFMLIB_INTEL_ATOM_FIXED2_ONLY,
	.pme_desc  = "Unhalted reference cycles. Measures bus cycles"
	},
/* 2 */{.pme_name  = "INSTRUCTIONS_RETIRED",
	.pme_code  = 0xc0,
	.pme_flags = PFMLIB_INTEL_ATOM_FIXED0|PFMLIB_INTEL_ATOM_PEBS,
	.pme_desc  = "Instructions retired"
	},
/* 3 */{.pme_name = "LAST_LEVEL_CACHE_REFERENCE",
	.pme_code = 0x4f2e,
	.pme_desc = "Last level of cache references"
	},
/* 4 */{.pme_name = "LAST_LEVEL_CACHE_MISSES",
	.pme_code = 0x412e,
	.pme_desc = "Last level of cache misses",
       },
/* 5  */{.pme_name = "BRANCH_INSTRUCTIONS_RETIRED",
	.pme_code  = 0xc4,
	.pme_desc  = "Branch instructions retired"
	},
/* 6  */{.pme_name = "MISPREDICTED_BRANCH_RETIRED",
	.pme_code  = 0xc5,
	.pme_desc  = "Mispredicted branch instruction retired"
	},
	
	/*
	 * BEGIN non architectural events
	 */
};
#define PME_INTEL_ATOM_UNHALTED_INTEL_ATOM_CYCLES	0
#define PME_INTEL_ATOM_INSTRUCTIONS_RETIRED		2
#define PME_INTEL_ATOM_EVENT_COUNT	   		(sizeof(intel_atom_pe)/sizeof(pme_intel_atom_entry_t))


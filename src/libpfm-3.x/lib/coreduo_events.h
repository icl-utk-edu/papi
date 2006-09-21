/*
 * Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
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

static pme_gen_ia32_entry_t coreduo_pe[]={
	/*
	 * BEGIN architectural perfmon events
	 */
/* 0 */{.pme_name    = "UNHALTED_CORE_CYCLES",
	.pme_code    = 0x003c,
	.pme_desc    = "Unhalted core cycles",
       },
/* 1 */{.pme_name = "UNHALTED_REFERENCE_CYCLES",
	.pme_code = 0x013c,
	.pme_desc = "Unhalted reference cycles. Measures bus cycles"
	},
/* 2 */{.pme_name = "INSTRUCTIONS_RETIRED",
	.pme_code = 0xc0,
	.pme_desc = "Instructions retired"
	},
/* 3 */{.pme_name = "LAST_LEVEL_CACHE_REFERENCE",
	.pme_code = 0x4f2e,
	.pme_desc = "Last level of cache references"
	},
/* 4 */{.pme_name    = "LAST_LEVEL_CACHE_MISSES",
	.pme_code    = 0x4120,
	.pme_desc    = "Last level of cache misses",
       },
/* 5  */{.pme_name = "BRANCH_INSTRUCTIONS_RETIRED",
	.pme_code = 0xc4,
	.pme_desc = "Branch instructions retired"
	},
/* 6  */{.pme_name = "MISPREDICTED_BRANCH_RETIRED",
	.pme_code = 0xc5,
	.pme_desc = "Mispredicted branch instruction retired"
	},

	/*
	 * BEGIN non architectural events
	 */

/* 7  */{.pme_name = "LD_BLOCKS",
	.pme_code = 0x03,
	.pme_desc = "Load operations delayed due to store buffer blocks. The preceding store may be blocked due to unknown address, unknown data, or conflict due to partial overlap between the load and store."
	},
/* 8  */{.pme_name = "SD_DRAINS",
	.pme_code = 0x04,
	.pme_desc = "Cycles while draining store buffers"
	},
/* 9  */{.pme_name = "MISALIGN_MEM_REF",
	.pme_code = 0x05,
	.pme_desc = "Misaligned data memory references (MOB splits of loads and stores)"
	},
/* 10 */{.pme_name = "SEG_REG_LOADS",
	.pme_code = 0x06,
	.pme_desc = "Segment register loads"
	},
/* 11 */{.pme_name = "SSE_PREFETCH",
	.pme_code = 0x7,
	.pme_desc = "SSE software prefetch instruction",
	.pme_flags   = PFMLIB_GEN_IA32_UMASK_COMBO,
	.pme_numasks = 3,
	.pme_umasks  = {
		{ .pme_uname = "NTA",
		  .pme_udesc = "prefetchnta retired",
		  .pme_ucode = 0x00,
		},
		{ .pme_uname = "T1",
		  .pme_udesc = "prefetcht1 retired",
		  .pme_ucode = 0x01,
		},
		{ .pme_uname = "T2",
		  .pme_udesc = "prefetcht2 retired",
		  .pme_ucode = 0x02,
		}
	 }
	},
/* 12 */{.pme_name = "SSE_NT_STORES_RETIRED",
	.pme_code = 0x0307,
	.pme_desc = "SSE streaming store instructions retired"
	},
/* 13 */{.pme_name = "L2_ADS",
	.pme_code = 0x21,
	.pme_desc = "L2 address stobes",
	.pme_numasks = 2,
	.pme_umasks  = {
		{ .pme_uname = "ALL_CORES",
		  .pme_udesc = "monitor all cores",
		  .pme_ucode = 0x3 << 6, /* 6=14-8, umask relative bit position */
		},
		{ .pme_uname = "THIS_CORE",
		  .pme_udesc = "monitor this core",
		  .pme_ucode = 0x1 << 6,
		},
	 }
	},
/* 14 */{.pme_name = "DBUS_BUSY",
	.pme_code = 0x22,
	.pme_desc = "Core cycle during which data bus was busy (increments by 4)",
	.pme_numasks = 2,
	.pme_umasks  = {
		{ .pme_uname = "ALL_CORES",
		  .pme_udesc = "monitoring all cores",
		  .pme_ucode = 0x3 << 6, /* 6=14-8, umask relative bit position */
		},
		{ .pme_uname = "THIS_CORE",
		  .pme_udesc = "monitoring this core",
		  .pme_ucode = 0x1 << 6,
		},
	 }
	},
/* 15 */{.pme_name = "L2_LINES_IN",
	.pme_code = 0x24,
	.pme_desc = "L2 cache lines allocated",
	.pme_numasks = 6,
	.pme_umasks  = {
		{ .pme_uname = "ALL_PREFETCHES_ALL_CORES",
		  .pme_udesc = "monitor all types of prefetches on all cores",
		  .pme_ucode = (0x3<<4)|(0x3<<6),
		},
		{ .pme_uname = "ALL_PREFETCHES_THIS_CORE",
		  .pme_udesc = "monitor all types of prefetches on this core",
		  .pme_ucode = (0x3<<4)|(0x1<<6),
		},
		{ .pme_uname = "HW_PREFETCH_ONLY_ALL_CORES",
		  .pme_udesc = "monitor only hardware prefetches on all cores",
		  .pme_ucode = (0x1<<4)|(0x3<<6),
		},
		{ .pme_uname = "HW_PREFETCH_ONLY_THIS_CORE",
		  .pme_udesc = "monitor only hardware prefetches on this core",
		  .pme_ucode = (0x1<<4)|(0x1<<6),
		},
		{ .pme_uname = "EXCL_HW_PREFETCH_ALL_CORES",
		  .pme_udesc = "monitoring exclude hardware prefetches on all cores",
		  .pme_ucode = 0x3<<6,
		},
		{ .pme_uname = "EXCL_HW_PREFETCH_THIS_CORE",
		  .pme_udesc = "monitoring exclude hardware prefetches on this core",
		  .pme_ucode = 0x1<<6,
		}
	 }
	},
/* 16 */{.pme_name = "BUS_DRDY_CLOCKS",
	.pme_code = 0x62,
	.pme_desc = "External bus cycles while DRDY is asserted",
	.pme_numasks = 2,
	.pme_umasks  = {
		{ .pme_uname = "ALL_AGENTS",
		  .pme_udesc = "monitoring all agents",
		  .pme_ucode = 0x1<<5, /* 5=13-8, umask relative bit position */
		},
		{ .pme_uname = "THIS_AGENT",
		  .pme_udesc = "monitoring this agent",
		  .pme_ucode = 0x0,
		},
	 }
	},
/* 17 */{.pme_name = "BUS_TRANS_RFO",
	.pme_code = 0x66,
	.pme_desc = "Completed read for ownership (RFO) transactions",
	.pme_numasks = 4,
	.pme_umasks  = {
		{ .pme_uname = "ALL_CORES_ALL_AGENTS",
		  .pme_udesc = "monitoring all cores and all agents",
		  .pme_ucode = (0x3<<6)|(0x1<<5),
		},
		{ .pme_uname = "ALL_CORES_THIS_AGENT",
		  .pme_udesc = "monitoring all cores and this agent",
		  .pme_ucode = 0x3<<6,
		},
		{ .pme_uname = "THIS_CORE_ALL_AGENTS",
		  .pme_udesc = "monitoring this core and all agents",
		  .pme_ucode = (0x1<<6)|(0x1<<5),
		},
		{ .pme_uname = "THIS_CORE_THIS_AGENT",
		  .pme_udesc = "monitoring this core and this agent",
		  .pme_ucode = 0x1<<6,
		},
	 }
	}
};
#define PME_COREDUO_UNHALTED_CORE_CYCLES	0
#define PME_COREDUO_INSTRUCTIONS_RETIRED	2
#define PME_COREDUO_EVENT_COUNT	   (sizeof(coreduo_pe)/sizeof(pme_gen_ia32_entry_t))

/*
 * Copyright (c) 2007 TOSHIBA CORPORATION based on code from
 * Copyright (c) 2001-2006 Hewlett-Packard Development Company, L.P.
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
 */

static pme_cell_entry_t cell_pe[] = {
	{.pme_name = "Branch_Commit",
	 .pme_desc = " Branch instruction committed.  ",
	 .pme_code = 0x834,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_OCCURRENCE,
	},
	{.pme_name = "Branch_Flush",
	 .pme_desc = " Branch instruction that caused a misprediction flush is committed. Branch misprediction includes: (1) misprediction of taken or not-taken on conditional branch, (2) misprediction of branch target address on bclr[1] and bcctr[1].  ",
	 .pme_code = 0x835,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_OCCURRENCE,
	},
	{.pme_name = "Ibuf_Empty",
	 .pme_desc = " Instruction buffer empty.  ",
	 .pme_code = 0x836,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_CUMULATIVE_LEN,
	},
	{.pme_name = "IERAT_Miss",
	 .pme_desc = " Instruction effective-address-to-real-address translation (I-ERAT) miss.  ",
	 .pme_code = 0x837,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_OCCURRENCE,
	},
	{.pme_name = "IL1_Miss_Cycles",
	 .pme_desc = " : L1 Instruction cache miss cycles. Counts the cycles from the miss event until the returned instruction is dispatched or cancelled due to branch misprediction, completion restart, or exceptions (see Note 1).  ",
	 .pme_code = 0x838,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_BOTH_TYPE,
	},
	{.pme_name = "Dispatch_Blocked",
	 .pme_desc = " : Valid instruction available for dispatch, but dispatch is blocked. ",
	 .pme_code = 0x83a,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_CUMULATIVE_LEN,
	},
	{.pme_name = "Instr_Flushed",
	 .pme_desc = " Instruction in pipeline stage EX7 causes a flush.  ",
	 .pme_code = 0x83d,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_OCCURRENCE,
	},
	{.pme_name = "PPC_Commit",
	 .pme_desc = " Two PowerPC instructions committed. For microcode sequences, only the last microcode operation is counted. Committed instructions are counted two at a time. If only one instruction has committed for a given cycle, this event will not be raised until another instruction has been committed in a future cycle.  ",
	 .pme_code = 0x83f,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_OCCURRENCE,
	},
	{.pme_name = "23_2",
	 .pme_desc = "  Data effective-address-to-real-address translation (D-ERAT) miss. This event is not speculative.    ",
	 .pme_code = 0x8fe,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_CUMULATIVE_LEN,
	},
	{.pme_name = "23_3",
	 .pme_desc = "  Store request counted at the L2 interface. This counts microcoded PowerPC Processor Element (PPE) sequences more than once (see Note 1 for exceptions).   ",
	 .pme_code = 0x8ff,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_CUMULATIVE_LEN,
	},
	{.pme_name = "23_4",
	 .pme_desc = "  Load valid at a particular pipe stage. This is speculative because flushed operations are also counted. Counts microcoded PPE sequences more than once. Misaligned flushes might be counted the first time as well. Load operations include all loads that read data from the cache, dcbt and dcbtst. This event does not include load Vector/single instruction multiple data (SIMD) multimedia extension pattern instructions.   ",
	 .pme_code = 0x900,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_CUMULATIVE_LEN,
	},
	{.pme_name = "23_5",
	 .pme_desc = "  L1 D-cache load miss. Pulsed when there is a miss request that has a tag miss but not an effective-address-to-real-address translation (ERAT) miss. This is speculative because flushed operations are counted as well.   ",
	 .pme_code = 0x901,
	 .pme_enable_word = WORD_0_AND_1,
	 .pme_freq = PFM_CELL_PME_FREQ_PPU_MFC,
	 .pme_type = COUNT_TYPE_CUMULATIVE_LEN,
	},
};
#define PME_CELL_EVENT_COUNT	(sizeof(cell_pe)/sizeof(pme_cell_entry_t))

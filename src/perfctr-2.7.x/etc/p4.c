/* $Id: p4.c,v 1.7 2004/02/20 21:33:25 mikpe Exp $
 *
 * pipe stdout through 'sort -u' to see:
 * - which ESCRs are usable, and the events they support
 * - which COUNTERs/CCCRs are usable, and the usable ESCRs they support
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define ARRAY_SIZE(A)	(sizeof(A)/sizeof((A)[0]))

/*
 * The 18 40-bit counters.
 */

struct counter {
    const char name[16];
};

static const struct counter counters[18] = {
/*   nr    name			   available ESCRs */
    [ 0] { "BPU_COUNTER0" },	/* {BPU,BSU,FSB,ITLB,MOB,PMH}_ESCR0 */
    [ 1] { "BPU_COUNTER1" },	/* {BPU,BSU,FSB,ITLB,MOB,PMH}_ESCR0 */
    [ 2] { "BPU_COUNTER2" },	/* {BPU,BSU,FSB,ITLB,MOB,PMH}_ESCR1 */
    [ 3] { "BPU_COUNTER3" },	/* {BPU,BSU,FSB,ITLB,MOB,PMH}_ESCR1 */
    [ 4] { "MS_COUNTER0" },	/* {MS,TBPU,TC}_ESCR0 */
    [ 5] { "MS_COUNTER1" },	/* {MS,TBPU,TC}_ESCR0 */
    [ 6] { "MS_COUNTER2" },	/* {MS,TBPU,TC}_ESCR1 */
    [ 7] { "MS_COUNTER3" },	/* {MS,TBPU,TC}_ESCR1 */
    [ 8] { "FLAME_COUNTER0" },	/* {DAC,FIRM,SAAT}_ESCR0 */
    [ 9] { "FLAME_COUNTER1" },	/* {DAC,FIRM,SAAT}_ESCR0 */
    [10] { "FLAME_COUNTER2" },	/* {DAC,FIRM,SAAT}_ESCR1 */
    [11] { "FLAME_COUNTER3" },	/* {DAC,FIRM,SAAT}_ESCR1 */
    [12] { "IQ_COUNTER0" },	/* ALF_ESCR0, CRU_ESCR0, CRU_ESCR2, RAT_ESCR0 */
    [13] { "IQ_COUNTER1" },	/* ALF_ESCR0, CRU_ESCR0, CRU_ESCR2, RAT_ESCR0 */
    [14] { "IQ_COUNTER2" },	/* ALF_ESCR1, CRU_ESCR1, CRU_ESCR3, RAT_ESCR1 */
    [15] { "IQ_COUNTER3" },	/* ALF_ESCR1, CRU_ESCR1, CRU_ESCR3, RAT_ESCR1 */
    [16] { "IQ_COUNTER4" },	/* ALF_ESCR0, CRU_ESCR0, CRU_ESCR2, RAT_ESCR0 */
    [17] { "IQ_COUNTER5" },	/* ALF_ESCR0, CRU_ESCR1, CRU_ESCR3, RAT_ESCR1 */
};

static unsigned int counter_msr(unsigned int counter_num)
{
    assert(counter_num < ARRAY_SIZE(counters));
    return 0x300 + counter_num;
}

static const char *counter_name(unsigned int counter_num)
{
    assert(counter_num < ARRAY_SIZE(counters));
    return counters[counter_num].name;
}

/*
 * The 18 counter configuration control registers (CCCRs).
 * They are in a one-to-one relation with the counters.
 */

struct cccr {
    const char name[16];
};

static const struct cccr cccrs[18] = {
    [ 0] { "BPU_CCCR0" },
    [ 1] { "BPU_CCCR1" },
    [ 2] { "BPU_CCCR2" },
    [ 3] { "BPU_CCCR3" },
    [ 4] { "MS_CCCR0" },
    [ 5] { "MS_CCCR1" },
    [ 6] { "MS_CCCR2" },
    [ 7] { "MS_CCCR3" },
    [ 8] { "FLAME_CCCR0" },
    [ 9] { "FLAME_CCCR1" },
    [10] { "FLAME_CCCR2" },
    [11] { "FLAME_CCCR3" },
    [12] { "IQ_CCCR0" },
    [13] { "IQ_CCCR1" },
    [14] { "IQ_CCCR2" },
    [15] { "IQ_CCCR3" },
    [16] { "IQ_CCCR4" },
    [17] { "IQ_CCCR5" },
};

static unsigned int cccr_msr(unsigned int cccr_num)
{
    assert(cccr_num < ARRAY_SIZE(cccrs));
    return 0x360 + cccr_num;
}

static const char *cccr_name(unsigned int cccr_num)
{
    assert(cccr_num < ARRAY_SIZE(cccrs));
    return cccrs[cccr_num].name;
}

/*
 * The 45 event selection control registers (ESCRs).
 */

enum escr_num {
    BSU_ESCR0,	/* BSQ_allocation, BSQ_cache_reference */
    BSU_ESCR1,	/* bsq_active_entries, BSQ_cache_reference */
    FSB_ESCR0,	/* FSB_data_activity, IOQ_allocation, b2b_cycles, bnr, global_power_events, response, snoopt */
    FSB_ESCR1,	/* FSB_data_activity, IOQ_active_entries, IOQ_allocation, b2b_cycles, bnr, global_power_events, response, snoop */
    FIRM_ESCR0,	/* 128bit_MMX_uop, 64bit_MMX_uop, SSE_input_assist, packed_DP_uop, packed_SP_uop, scalar_DP_uop, scalar_SP_uop, x87_FP_uop, x86_SIMD_moves_uop */
    FIRM_ESCR1,	/* 128bit_MMX_uop, 64bit_MMX_uop, SSE_input_assist, packed_DP_uop, packed_SP_uop, scalar_DP_uop, scalar_SP_uop, x87_FP_uop, x86_SIMD_moves_uop */
    FLAME_ESCR0,/* UNUSED */
    FLAME_ESCR1,/* UNUSED */
    DAC_ESCR0,	/* WC_Buffer, memory_cancel */
    DAC_ESCR1,	/* WC_Buffer, memory_cancel */
    MOB_ESCR0,	/* MOB_load_replay */
    MOB_ESCR1,	/* MOB_load_replay */
    PMH_ESCR0,	/* page_walk_type */
    PMH_ESCR1,	/* page_walk_type */
    SAAT_ESCR0,	/* load_port_replay, memory_complete, store_port_replay */
    SAAT_ESCR1,	/* load_port_replay, memory_complete, store_port_replay */
    U2L_ESCR0,	/* UNUSED */
    U2L_ESCR1,	/* UNUSED */
    BPU_ESCR0,	/* BPU_fetch_request */
    BPU_ESCR1,	/* BPU_fetch_request */
    IS_ESCR0,	/* UNUSED */
    IS_ESCR1,	/* UNUSED */
    ITLB_ESCR0,	/* ITLB_reference */
    ITLB_ESCR1,	/* ITLB_reference */
    CRU_ESCR0,	/* instr_retired, mispred_branch_retired, uops_retired, instr_completed */
    CRU_ESCR1,	/* instr_retired, mispred_branch_retired, uops_retired, instr_completed */
    IQ_ESCR0,	/* UNUSED; available in family 0x0F models 1 and 2, removed from later models */
    IQ_ESCR1,	/* UNUSED; available in family 0x0F models 1 and 2, removed from later models */
    RAT_ESCR0,	/* uop_type */
    RAT_ESCR1,	/* uop_type */
    SSU_ESCR0,	/* UNUSED */
    MS_ESCR0,	/* tc_ms_xfer, uop_queue_writes */
    MS_ESCR1,	/* tc_ms_xfer, uop_queue_writes */
    TBPU_ESCR0,	/* retired_branch_type, retired_mispred_branch_type */
    TBPU_ESCR1,	/* retired_branch_type, retired_mispred_branch_type */
    TC_ESCR0,	/* TC_deliver_mode, TC_misc */
    TC_ESCR1,	/* TC_deliver_mode, TC_misc */
    IX_ESCR0,	/* UNUSED */
    IX_ESCR1,	/* UNUSED */
    ALF_ESCR0,	/* resource_stall */
    ALF_ESCR1,	/* resource_stall */
    CRU_ESCR2,	/* branch_retired, execution_event, front_end_event, machine_clear, replay_event, x87_assist */
    CRU_ESCR3,	/* branch_retired, execution_event, front_end_event, machine_clear, replay_event, x87_assist */
    CRU_ESCR4,	/* UNUSED */
    CRU_ESCR5,	/* UNUSED */
};

struct escr {
    const char name[16];
};

static const struct escr escrs[45] = {
    [BSU_ESCR0] { "BSU_ESCR0" },
    [BSU_ESCR1] { "BSU_ESCR1" },
    [FSB_ESCR0] { "FSB_ESCR0" },
    [FSB_ESCR1] { "FSB_ESCR1" },
    [FIRM_ESCR0] { "FIRM_ESCR0" },
    [FIRM_ESCR1] { "FIRM_ESCR1" },
    [FLAME_ESCR0] { "FLAME_ESCR0" },
    [FLAME_ESCR1] { "FLAME_ESCR1" },
    [DAC_ESCR0] { "DAC_ESCR0" },
    [DAC_ESCR1] { "DAC_ESCR1" },
    [MOB_ESCR0] { "MOB_ESCR0" },
    [MOB_ESCR1] { "MOB_ESCR1" },
    [PMH_ESCR0] { "PMH_ESCR0" },
    [PMH_ESCR1] { "PMH_ESCR1" },
    [SAAT_ESCR0] { "SAAT_ESCR0" },
    [SAAT_ESCR1] { "SAAT_ESCR1" },
    [U2L_ESCR0] { "U2L_ESCR0" },
    [U2L_ESCR1] { "U2L_ESCR1" },
    [BPU_ESCR0] { "BPU_ESCR0" },
    [BPU_ESCR1] { "BPU_ESCR1" },
    [IS_ESCR0] { "IS_ESCR0" },
    [IS_ESCR1] { "IS_ESCR1" },
    [ITLB_ESCR0] { "ITLB_ESCR0" },
    [ITLB_ESCR1] { "ITLB_ESCR1" },
    [CRU_ESCR0] { "CRU_ESCR0" },
    [CRU_ESCR1] { "CRU_ESCR1" },
    [IQ_ESCR0] { "IQ_ESCR0" },
    [IQ_ESCR1] { "IQ_ESCR1" },
    [RAT_ESCR0] { "RAT_ESCR0" },
    [RAT_ESCR1] { "RAT_ESCR1" },
    [SSU_ESCR0] { "SSU_ESCR0" },
    [MS_ESCR0] { "MS_ESCR0" },
    [MS_ESCR1] { "MS_ESCR1" },
    [TBPU_ESCR0] { "TBPU_ESCR0" },
    [TBPU_ESCR1] { "TBPU_ESCR1" },
    [TC_ESCR0] { "TC_ESCR0" },
    [TC_ESCR1] { "TC_ESCR1" },
    [IX_ESCR0] { "IX_ESCR0" },
    [IX_ESCR1] { "IX_ESCR1" },
    [ALF_ESCR0] { "ALF_ESCR0" },
    [ALF_ESCR1] { "ALF_ESCR1" },
    [CRU_ESCR2] { "CRU_ESCR2" },
    [CRU_ESCR3] { "CRU_ESCR3" },
    [CRU_ESCR4] { "CRU_ESCR4" },
    [CRU_ESCR5] { "CRU_ESCR5" },
};

static unsigned int escr_msr(enum escr_num escr_num)
{
    assert(escr_num < ARRAY_SIZE(escrs));
    if( escr_num >= CRU_ESCR4 )
	return 0x3E0 + (escr_num - CRU_ESCR4);
    if( escr_num >= IX_ESCR0 )
	return 0x3C8 + (escr_num - IX_ESCR0);
    if( escr_num >= MS_ESCR0 )
	return 0x3C0 + (escr_num - MS_ESCR0);
    return 0x3A0 + escr_num;
}

static const char *escr_name(enum escr_num escr_num)
{
    assert(escr_num < ARRAY_SIZE(escrs));
    return escrs[escr_num].name;
}

/*
 * The map from CCCR number and ESCR select value to ESCR MSR address.
 * This is the manual's original uncompacted table.
 */

static const unsigned short p4_cccr_escr_map_orig[18][8] = {
     [0x00] {		[7] 0x3A0,
			[6] 0x3A2,
			[2] 0x3AA,
			[4] 0x3AC,
			[0] 0x3B2,
			[1] 0x3B4,
			[3] 0x3B6,
			[5] 0x3C8, },
     [0x01] {		[7] 0x3A0,
			[6] 0x3A2,
			[2] 0x3AA,
			[4] 0x3AC,
			[0] 0x3B2,
			[1] 0x3B4,
			[3] 0x3B6,
			[5] 0x3C8, },
     [0x02] {		[7] 0x3A1,
			[6] 0x3A3,
			[2] 0x3AB,
			[4] 0x3AD,
			[0] 0x3B3,
			[1] 0x3B5,
			[3] 0x3B7,
			[5] 0x3C9, },
     [0x03] {		[7] 0x3A1,
			[6] 0x3A3,
			[2] 0x3AB,
			[4] 0x3AD,
			[0] 0x3B3,
			[1] 0x3B5,
			[3] 0x3B7,
			[5] 0x3C9, },
     [0x04] {		[0] 0x3C0,
			[2] 0x3C2,
			[1] 0x3C4, },
     [0x05] {		[0] 0x3C0,
			[2] 0x3C2,
			[1] 0x3C4, },
     [0x06] {		[0] 0x3C1,
			[2] 0x3C3,
			[1] 0x3C5, },
     [0x07] {		[0] 0x3C1,
			[2] 0x3C3,
			[1] 0x3C5, },
     [0x08] {		[1] 0x3A4,
			[0] 0x3A6,
			[5] 0x3A8,
			[2] 0x3AE,
			[3] 0x3B0, },
     [0x09] {		[1] 0x3A4,
			[0] 0x3A6,
			[5] 0x3A8,
			[2] 0x3AE,
			[3] 0x3B0, },
     [0x0A] {		[1] 0x3A5,
			[0] 0x3A7,
			[5] 0x3A9,
			[2] 0x3AF,
			[3] 0x3B1, },
     [0x0B] {		[1] 0x3A5,
			[0] 0x3A7,
			[5] 0x3A9,
			[2] 0x3AF,
			[3] 0x3B1, },
     [0x0C] {		[4] 0x3B8,
			[5] 0x3CC,
			[6] 0x3E0,
			[0] 0x3BA,
			[2] 0x3BC,
			[3] 0x3BE,
			[1] 0x3CA, },
     [0x0D] {		[4] 0x3B8,
			[5] 0x3CC,
			[6] 0x3E0,
			[0] 0x3BA,
			[2] 0x3BC,
			[3] 0x3BE,
			[1] 0x3CA, },
     [0x0E] {		[4] 0x3B9,
			[5] 0x3CD,
			[6] 0x3E1,
			[0] 0x3BB,
			[2] 0x3BD,
			[1] 0x3CB, },
     [0x0F] {		[4] 0x3B9,
			[5] 0x3CD,
			[6] 0x3E1,
			[0] 0x3BB,
			[2] 0x3BD,
			[1] 0x3CB, },
     [0x10] {		[4] 0x3B8,
			[5] 0x3CC,
			[6] 0x3E0,
			[0] 0x3BA,
			[2] 0x3BC,
			[3] 0x3BE,
			[1] 0x3CA, },
     [0x11] {		[4] 0x3B9,
			[5] 0x3CD,
			[6] 0x3E1,
			[0] 0x3BB,
			[2] 0x3BD,
			[1] 0x3CB, },
};

static unsigned int p4_escr_addr_orig(unsigned int pmc, unsigned int escr_select)
{
     if( pmc > 0x11 || escr_select > 7 )
	  return 0;
     return p4_cccr_escr_map_orig[pmc][escr_select];
};

/*
 * The map from CCCR number and ESCR select value to ESCR MSR address.
 * This is the compacted map, derived from the manual's table.
 */

static const unsigned char p4_cccr_escr_map[4][8] = {
	/* 0x00 and 0x01 as is, 0x02 and 0x03 are +1 */
	[0x00/4] {	[7] 0xA0,
			[6] 0xA2,
			[2] 0xAA,
			[4] 0xAC,
			[0] 0xB2,
			[1] 0xB4,
			[3] 0xB6,
			[5] 0xC8, },
	/* 0x04 and 0x05 as is, 0x06 and 0x07 are +1 */
	[0x04/4] {	[0] 0xC0,
			[2] 0xC2,
			[1] 0xC4, },
	/* 0x08 and 0x09 as is, 0x0A and 0x0B are +1 */
	[0x08/4] {	[1] 0xA4,
			[0] 0xA6,
			[5] 0xA8,
			[2] 0xAE,
			[3] 0xB0, },
	/* 0x0C, 0x0D, and 0x10 as is,
	   0x0E, 0x0F, and 0x11 are +1 except [3] is not in the domain */
	[0x0C/4] {	[4] 0xB8,
			[5] 0xCC,
			[6] 0xE0,
			[0] 0xBA,
			[2] 0xBC,
			[3] 0xBE,
			[1] 0xCA, },
};

static unsigned int p4_escr_addr(unsigned int pmc, unsigned int escr_select)
{
	unsigned int pair, escr_offset;

	if( pmc > 0x11 )
		return 0;	/* pmc range error */
	if( pmc > 0x0F )
		pmc -= 3;	/* 0 <= pmc <= 0x0F */
	pair = pmc / 2;		/* 0 <= pair <= 7 */
	escr_offset = p4_cccr_escr_map[pair / 2][escr_select];
	if( !escr_offset || (pair == 7 && escr_select == 3) )
		return 0;	/* ESCR SELECT range error */
	return escr_offset + (pair & 1) + 0x300;
};

static void check_p4_escr_addr(void)
{
     unsigned int pmc, escr_select, escr_addr_orig, escr_addr;

     for(pmc = 0; pmc < 0x12; ++pmc) {
	  for(escr_select = 0; escr_select < 8; ++escr_select) {
	       escr_addr_orig = p4_escr_addr_orig(pmc, escr_select);
	       escr_addr = p4_escr_addr(pmc, escr_select);
	       if( escr_addr_orig != escr_addr )
		    printf("p4_escr_addr(%u, %u) is 0x%03x, should be 0x%03x\n",
			   pmc, escr_select, escr_addr, escr_addr_orig);
	  }
     }
}

/*
 * The events.
 */

struct event {
    const char name[32];
    unsigned int select;	/* ESCR[31:25] */
    enum escr_num escr0;
    unsigned int escr1;		/* escr_num or -1 */
};

static const struct event events[] = {
    /* Non-Retirement Events: */
    { "TC_deliver_mode", 0x01, TC_ESCR0, TC_ESCR1 },
    { "BPU_fetch_request", 0x03, BPU_ESCR0, BPU_ESCR1 },
    { "ITLB_reference", 0x18, ITLB_ESCR0, ITLB_ESCR1 },
    { "memory_cancel", 0x02, DAC_ESCR0, DAC_ESCR1 },
    { "memory_complete", 0x08, SAAT_ESCR0, SAAT_ESCR1 },
    { "load_port_replay", 0x04, SAAT_ESCR0, SAAT_ESCR1 },
    { "store_port_replay", 0x05, SAAT_ESCR0, SAAT_ESCR1 },
    { "MOB_load_replay", 0x03, MOB_ESCR0, MOB_ESCR1 },
    { "page_walk_type", 0x01, PMH_ESCR0, PMH_ESCR1 },
    { "BSQ_cache_reference", 0x0C, BSU_ESCR0, BSU_ESCR1 },
    { "IOQ_allocation", 0x03, FSB_ESCR0, FSB_ESCR1 }, /* ESCR1 unavailable if CPUID < 0xF27 */
    { "IOQ_active_entries", 0x1A, FSB_ESCR1, -1 },
    { "FSB_data_activity", 0x17, FSB_ESCR0, FSB_ESCR1 },
    { "BSQ_allocation", 0x05, BSU_ESCR0, -1 },
    { "bsq_active_entries", 0x06, BSU_ESCR1, -1 },
    { "SSE_input_assist", 0x34, FIRM_ESCR0, FIRM_ESCR1 },
    { "packed_SP_uop", 0x08, FIRM_ESCR0, FIRM_ESCR1 },
    { "packed_DP_uop", 0x0C, FIRM_ESCR0, FIRM_ESCR1 },
    { "scalar_SP_uop", 0x0A, FIRM_ESCR0, FIRM_ESCR1 },
    { "scalar_DP_uop", 0x0E, FIRM_ESCR0, FIRM_ESCR1 },
    { "64bit_MMX_uop", 0x02, FIRM_ESCR0, FIRM_ESCR1 },
    { "128bit_MMX_uop", 0x1A, FIRM_ESCR0, FIRM_ESCR1 },
    { "x87_FP_uop", 0x04, FIRM_ESCR0, FIRM_ESCR1 },
    { "x87_SIMD_moves_uop", 0x2E, FIRM_ESCR0, FIRM_ESCR1 },
    { "TC_misc", 0x06, TC_ESCR0, TC_ESCR1 },
    { "global_power_events", 0x13, FSB_ESCR0, FSB_ESCR1 },
    { "tc_ms_xfer", 0x05, MS_ESCR0, MS_ESCR1 },
    { "uop_queue_writes", 0x09, MS_ESCR0, MS_ESCR1 },
    { "retired_mispred_branch_type", 0x05, TBPU_ESCR0, TBPU_ESCR1 },
    { "retired_branch_type", 0x04, TBPU_ESCR0, TBPU_ESCR1 },
    { "resource_stall", 0x01, ALF_ESCR0, ALF_ESCR1 },
    { "WC_Buffer", 0x05, DAC_ESCR0, DAC_ESCR1 },
    { "b2b_cycles", 0x16, FSB_ESCR0, FSB_ESCR1 },
    { "bnr", 0x08, FSB_ESCR0, FSB_ESCR1 },
    { "snoop", 0x06, FSB_ESCR0, FSB_ESCR1 },
    { "response", 0x04, FSB_ESCR0, FSB_ESCR1 },
    /* At-Retirement Events: */
    { "front_end_event", 0x08, CRU_ESCR2, CRU_ESCR3 }, /* filters uop_type */
    { "execution_event", 0x0C, CRU_ESCR2, CRU_ESCR3 }, /* filters packed_SP_uop, packed_DP_uop, scalar_SP_uop, scalar_DP_uop, 128bit_MMX_uop, 64bit_MMX_uop, x87_FP_uop, x86_SIMD_moves_uop */
    { "replay_event", 0x09, CRU_ESCR2, CRU_ESCR3 }, /* filters MOB_load_replay, load_port_replay(SAAT_ESCR1), store_port_replay(SAAT_ESCR0), MSR_IA32_PEBS_ENABLE, MSR_PEBS_MATRIX_VERT */
    { "instr_retired", 0x02, CRU_ESCR0, CRU_ESCR1 }, /* seems to be sensitive to tagged uops */
    { "uops_retired", 0x01, CRU_ESCR0, CRU_ESCR1 },
    { "uop_type", 0x02, RAT_ESCR0, RAT_ESCR1 }, /* can tag uops for front_end_event */
    { "branch_retired", 0x06, CRU_ESCR2, CRU_ESCR3 },
    { "mispred_branch_retired", 0x03, CRU_ESCR0, CRU_ESCR1 },
    { "x87_assist", 0x03, CRU_ESCR2, CRU_ESCR3 },
    { "machine_clear", 0x02, CRU_ESCR2, CRU_ESCR3 },
    /* Model 3 only */
    { "instr_completed", 0x07, CRU_ESCR0, CRU_ESCR1 },
};

static void do_escr(unsigned int escr_num)
{
     unsigned int pmc;
     unsigned int escr_select;
     unsigned int msr;
     const char *name;

     msr = escr_msr(escr_num);
     name = escr_name(escr_num);

     for(pmc = 0; pmc < ARRAY_SIZE(counters); ++pmc) {
	  for(escr_select = 0; escr_select < 8; ++escr_select) {
	       if( p4_escr_addr(pmc, escr_select) == msr )
		    printf("counter %s escr %s\n", counter_name(pmc), name);
	  }
     }
}

static void do_events(void)
{
    unsigned int i;

    for(i = 0; i < ARRAY_SIZE(events); ++i) {
	const struct event *event = &events[i];
	printf("escr %s event %s\n",
	       escr_name(event->escr0),
	       event->name);
	do_escr(event->escr0);
	if( event->escr1 != -1 ) {
	    printf("escr %s event %s\n",
		   escr_name(event->escr1),
		   event->name);
	    do_escr(event->escr1);
	}
    }
}

int main(void)
{
    check_p4_escr_addr();
    do_events();
    return 0;
}

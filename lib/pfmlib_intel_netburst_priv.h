/*
 * Copyright (c) 2006 IBM Corp.
 * Contributed by Kevin Corry <kevcorry@us.ibm.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * pfmlib_netburst_priv.h
 *
 * Structures and definitions for use in the Pentium4/Xeon/EM64T libpfm code.
 */

#ifndef _PFMLIB_INTEL_NETBURST_PRIV_H_
#define _PFMLIB_INTEL_NETBURST_PRIV_H_

/* ESCR: Event Selection Control Register
 *
 * These registers are used to select which event to count along with options
 * for that event. There are (up to) 45 ESCRs, but each data counter is
 * restricted to a specific set of ESCRs.
 */

/**
 * netburst_escr_value_t
 *
 * Bit-wise breakdown of the ESCR registers.
 *
 *    Bits     Description
 *   -------   -----------
 *   63 - 31   Reserved
 *   30 - 25   Event Select
 *   24 - 9    Event Mask
 *    8 - 5    Tag Value
 *      4      Tag Enable
 *      3      T0 OS - Enable counting in kernel mode (thread 0)
 *      2      T0 USR - Enable counting in user mode (thread 0)
 *      1      T1 OS - Enable counting in kernel mode (thread 1)
 *      0      T1 USR - Enable counting in user mode (thread 1)
 **/

#define EVENT_MASK_BITS 16
#define EVENT_SELECT_BITS 6

typedef union {
	unsigned long val;
	struct {
		unsigned long t1_usr:1;
		unsigned long t1_os:1;
		unsigned long t0_usr:1;
		unsigned long t0_os:1;
		unsigned long tag_enable:1;
		unsigned long tag_value:4;
		unsigned long event_mask:EVENT_MASK_BITS;
		unsigned long event_select:EVENT_SELECT_BITS;
		unsigned long reserved:1;
	} bits;
} netburst_escr_value_t;

/* CCCR: Counter Configuration Control Register
 *
 * These registers are used to configure the data counters. There are 18
 * CCCRs, one for each data counter.
 */

/**
 * netburst_cccr_value_t
 *
 * Bit-wise breakdown of the CCCR registers.
 *
 *    Bits     Description
 *   -------   -----------
 *   63 - 32   Reserved
 *     31      OVF - The data counter overflowed.
 *     30      Cascade - Enable cascading of data counter when alternate
 *             counter overflows.
 *   29 - 28   Reserved
 *     27      OVF_PMI_T1 - Generate interrupt for LP1 on counter overflow
 *     26      OVF_PMI_T0 - Generate interrupt for LP0 on counter overflow
 *     25      FORCE_OVF - Force interrupt on every counter increment
 *     24      Edge - Enable rising edge detection of the threshold comparison
 *             output for filtering event counts.
 *   23 - 20   Threshold Value - Select the threshold value for comparing to
 *             incoming event counts.
 *     19      Complement - Select how incoming event count is compared with
 *             the threshold value.
 *     18      Compare - Enable filtering of event counts.
 *   17 - 16   Active Thread - Only used with HT enabled.
 *             00 - None: Count when neither LP is active.
 *             01 - Single: Count when only one LP is active.
 *             10 - Both: Count when both LPs are active.
 *             11 - Any: Count when either LP is active.
 *   15 - 13   ESCR Select - Select which ESCR to use for selecting the
 *             event to count.
 *     12      Enable - Turns the data counter on or off.
 *   11 - 0    Reserved
 **/
typedef union {
	unsigned long val;
	struct {
		unsigned long reserved1:12;
		unsigned long enable:1;
		unsigned long escr_select:3;
		unsigned long active_thread:2;
		unsigned long compare:1;
		unsigned long complement:1;
		unsigned long threshold:4;
		unsigned long edge:1;
		unsigned long force_ovf:1;
		unsigned long ovf_pmi_t0:1;
		unsigned long ovf_pmi_t1:1;
		unsigned long reserved2:2;
		unsigned long cascade:1;
		unsigned long overflow:1;
	} bits;
} netburst_cccr_value_t;


/**
 * netburst_escr_reg_t
 *
 * Describe one ESCR register.
 *
 * "netburst_escrs" is a flat array of these structures
 * that defines all the ESCRs.
 *
 * @name: ESCR's name
 * @pmc: Perfmon's PMC number for this ESCR.
 * @allowed_cccrs: Array of CCCR numbers that can be used with this ESCR. A
 *                 positive value is an index into the netburst_ccrs array.
 *                 A value of -1 indicates that slot is unused.
 **/

#define MAX_CCCRS_PER_ESCR 3

typedef struct {
	char *name;
	int pmc;
	int allowed_cccrs[MAX_CCCRS_PER_ESCR];
} netburst_escr_reg_t;


/* CCCR: Counter Configuration Control Register
 *
 * These registers are used to configure the data counters. There are 18
 * CCCRs, one for each data counter.
 */

/**
 * netburst_cccr_reg_t
 *
 * Describe one CCCR register.
 *
 * "netburst_cccrs" is a flat array of these structures
 * that defines all the CCCRs.
 *
 * @name: CCCR's name
 * @pmc: Perfmon's PMC number for this CCCR
 * @pmd: Perfmon's PMD number for the associated data counter. Every CCCR has
 *       exactly one counter.
 * @allowed_escrs: Array of ESCR numbers that can be used with this CCCR. A
 *                 positive value is an index into the netburst_escrs array.
 *                 A value of -1 indicates that slot is unused. The index into
 *                 this array is the value to use in the escr_select portion
 *                 of the CCCR value.
 **/

#define MAX_ESCRS_PER_CCCR 8

typedef struct {
	char *name;
	int pmc;
	int pmd;
	int allowed_escrs[MAX_ESCRS_PER_CCCR];
} netburst_cccr_reg_t;

/**
 * netburst_replay_regs_t
 *
 * Describe one pair of PEBS registers for use with the replay_event event.
 *
 * "p4_replay_regs" is a flat array of these structures
 * that defines all the PEBS pairs per Table A-10 of 
 * the Intel System Programming Guide Vol 3B.
 *
 * @enb:      value for the PEBS_ENABLE register for a given replay metric.
 * @mat_vert: value for the PEBS_MATRIX_VERT register for a given metric.
 *            The replay_event event defines a series of virtual mask bits
 *            that serve as indexes into this array. The values at that index
 *            provide information programmed into the PEBS registers to count
 *            specific metrics available to the replay_event event.
 **/

typedef struct {
	int enb;
	int mat_vert;
} netburst_replay_regs_t;

/**
 * netburst_pmc_t
 *
 * Provide a mapping from PMC number to the type of control register and
 * its index within the appropriate array.
 *
 * @name: Name
 * @type: NETBURST_PMC_TYPE_ESCR or NETBURST_PMC_TYPE_CCCR
 * @index: Index into the netburst_escrs array or the netburst_cccrs array.
 **/
typedef struct {
	char *name;
	int type;
	int index;
} netburst_pmc_t;

#define NETBURST_PMC_TYPE_ESCR 1
#define NETBURST_PMC_TYPE_CCCR 2

/**
 * netburst_event_mask_t
 *
 * Defines one bit of the event-mask for one Pentium4 event.
 *
 * @name: Event mask name
 * @desc: Event mask description
 * @bit: The bit position within the event_mask field.
 **/
typedef struct {
	char *name;
	char *desc;
	unsigned int bit;
} netburst_event_mask_t;

/**
 * netburst_event_t
 *
 * Describe one event that can be counted on Pentium4/EM64T.
 *
 * "netburst_events" is a flat array of these structures that defines
 * all possible events.
 *
 * @name: Event name
 * @desc: Event description
 * @event_select: Value for the 'event_select' field in the ESCR (bits [31:25]).
 * @escr_select: Value for the 'escr_select' field in the CCCR (bits [15:13]).
 * @allowed_escrs: Numbers for ESCRs that can be used to count this event. A
 *                 positive value is an index into the netburst_escrs array.
 *                 A value of -1 means that slot is not used.
 * @event_masks: Array of descriptions of available masks for this event.
 *               Array elements with a NULL 'name' field are unused.
 **/

#define MAX_ESCRS_PER_EVENT 2

typedef struct {
	char *name;
	char *desc;
	unsigned int event_select;
	unsigned int escr_select;
	int allowed_escrs[MAX_ESCRS_PER_EVENT];
	netburst_event_mask_t event_masks[EVENT_MASK_BITS];
} netburst_event_t;

#endif


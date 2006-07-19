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
 * applications on Linux/ia64.
 */
#ifndef __PFMLIB_GEN_IA32_PRIV_H__
#define __PFMLIB_GEN_IA32_PRIV_H__

/*
 * architected PMU encoding structure
 * (code must be first 8 bits)
 */
typedef struct {
	unsigned int pme_sel:8;		/* event select */
	unsigned int pme_umask:8;	/* unit mask */
	unsigned int pme_res:16;	/* reserved */
} pme_gen_ia32_entry_code_t;		

typedef union {
	pme_gen_ia32_entry_code_t pme_code;
	unsigned int pme_vcode;
} pme_gen_ia32_code_t;

typedef struct {
	char			*pme_name;
	pme_gen_ia32_code_t	pme_entry_code;
	char			*pme_desc;		/* text description of the event */
} pme_gen_ia32_entry_t;

typedef struct {
	unsigned int version:8;
	unsigned int num_cnt:8;
	unsigned int cnt_width:8;
	unsigned int ebx_length:8;
} pmu_eax_t;

typedef struct {
	unsigned int no_core_cycle:1;
	unsigned int no_inst_retired:1;
	unsigned int no_ref_cycle:1;
	unsigned int no_llc_ref:1;
	unsigned int no_llc_miss:1;
	unsigned int no_br_retired:1;
	unsigned int no_br_mispred_retired:1;
	unsigned int reserved:25;
} pmu_ebx_t;


#endif /* __PFMLIB_GEN_IA32_PRIV_H__ */

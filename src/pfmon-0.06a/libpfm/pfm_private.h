/*
 * 
 *
 * Copyright (C) 2001 Hewlett-Packard Co
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file is part of pfmon, a sample tool to measure performance 
 * of applications on Linux/ia64.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
 */
#ifndef __PFM_PRIVATE_H__
#define __PFM_PRIVATE_H__

/*
 * Data representation of the database used by the library
 *
 * The table used by the library is automatically generated. See makefile
 *
 * XXX: some fields could be collapsed to same save
 */

typedef union {
		unsigned int  pme_vcode;		/* virtual code: code+umask combined */
		struct		{
			unsigned int pme_mcode:8;	/* major event code */
			unsigned int pme_ear:1;		/* is EAR event */
			unsigned int pme_dear:1;	/* 1=Data 0=Instr */
			unsigned int pme_tlb:1;		/* 1=TLB 0=Cache */
			unsigned int pme_ig1:5;		/* ignored */
			unsigned int pme_umask:16;	/* unit mask*/
		} pme_codes;				/* event code divided in 2 parts */
	} pme_entry_code_t;				

typedef union {
		unsigned long qual;		/* generic qualifier */
		struct {
			unsigned long pme_iar:1;	/* instruction address range supported */
			unsigned long pme_opm:1;	/* opcode match supported */
			unsigned long pme_dar:1;	/* data address range supported */
		} pme_qual;
	} pme_qualifiers_t;

typedef struct pme_entry {
	char		 *pme_name;
	pme_entry_code_t pme_entry_code;	/* event code */
	unsigned int	 pme_thres;		/* maximum multi-occurence */
	unsigned long	 pme_counters;		/* supported counters */
	pme_qualifiers_t pme_qualifiers;	/* support qualifier */
} pme_entry_t;

#define PME_UMASK_NONE	0x0

/*
 * We embed the umask value into the event code. Because it really is 
 * like a subevent.
 * pme_code:
 * 	- lower 16 bits: major event code
 * 	- upper 16 bits: unit mask
 */
#define pme_vcode	pme_entry_code.pme_vcode
#define pme_code	pme_entry_code.pme_codes.pme_mcode
#define pme_ear		pme_entry_code.pme_codes.pme_ear
#define pme_dear	pme_entry_code.pme_codes.pme_dear
#define pme_tlb		pme_entry_code.pme_codes.pme_tlb
#define pme_umask	pme_entry_code.pme_codes.pme_umask
#define pme_used	pme_qualifiers.pme_qual_struct.pme_used

#define event_is_ear(e)	((e)->pme_ear == 1)
#define event_is_iear(e) ((e)->pme_ear == 1 && (e)->pme_dear==0)
#define event_is_dear(e) ((e)->pme_ear == 1 && (e)->pme_dear==1)
#define event_is_tlb_ear(e) ((e)->pme_ear == 1 && (e)->pme_tlb==1)
#define event_is_btb(e)		((e)->pme_code == pe[PME_BRANCH_EVENT].pme_code)

#define event_opcm_ok(e) ((e)->pme_qualifiers.pme_qual.pme_opm==1)
#define event_iar_ok(e)	 ((e)->pme_qualifiers.pme_qual.pme_iar==1)
#define event_dar_ok(e)	 ((e)->pme_qualifiers.pme_qual.pme_dar==1)

#endif /* __PFM_PRIVATE_H__ */


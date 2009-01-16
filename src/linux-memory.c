/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    linux-memory.c
* Author:  Kevin London
*          london@cs.utk.edu
* Mods:    Dan Terpstra
*          terpstra@eecs.utk.edu
*          complete rewrite to conform to latest docs and convert
*          Intel to a table driven implementation.
*          Now also supports multiple TLB descriptors
*/

#include "papi.h"
#include "papi_internal.h"

static void init_mem_hierarchy(PAPI_mh_info_t * mh_info);
static int init_amd(PAPI_mh_info_t * mh_info);
static short int _amd_L2_L3_assoc(unsigned short int pattern);
static int init_intel(PAPI_mh_info_t * mh_info);
inline_static void cpuid(unsigned int *, unsigned int *, unsigned int *, unsigned int *);

int _papi_hwd_get_memory_info(PAPI_hw_info_t * hw_info, int cpu_type)
{
	int retval = 0;
	union {
		struct {
			unsigned int ax, bx,cx,dx;
		} e;
		char vendor[20]; /* leave room for terminator bytes */
	} reg;

	/* Don't use cpu_type to determine the processor.
	 * get the information directly from the chip.
	 */
	reg.e.ax = 0; /* function code 0: vendor string */
	/* The vendor string is composed of EBX:EDX:ECX.
	 * by swapping the register addresses in the call below,
	 * the string is correctly composed in the char array.
	 */
	cpuid(&reg.e.ax, &reg.e.bx, &reg.e.dx, &reg.e.cx);
	reg.vendor[16] = 0; 
	MEMDBG("Vendor: %s\n", &reg.vendor[4]);

	init_mem_hierarchy(&hw_info->mem_hierarchy);

	if (!strncmp("GenuineIntel", &reg.vendor[4], 12)) {
		hw_info->mem_hierarchy.levels = init_intel(&hw_info->mem_hierarchy);
	} else if (strncmp("GenuineIntel", &reg.vendor[4], 12)) {
		hw_info->mem_hierarchy.levels = init_amd(&hw_info->mem_hierarchy);
	} else {
		MEMDBG("Unsupported cpu type; Not Intel or AMD x86\n");
		return(PAPI_ESBSTR);
	}

	/* This works only because an empty cache element is initialized to 0 */
	MEMDBG("Detected L1: %d L2: %d  L3: %d\n",
		hw_info->mem_hierarchy.level[0].cache[0].size + hw_info->mem_hierarchy.level[0].cache[1].size, 
		hw_info->mem_hierarchy.level[1].cache[0].size + hw_info->mem_hierarchy.level[1].cache[1].size, 
		hw_info->mem_hierarchy.level[2].cache[0].size + hw_info->mem_hierarchy.level[2].cache[1].size);
	return retval;
}

static void init_mem_hierarchy(PAPI_mh_info_t * mh_info) {
	int i,j;
	PAPI_mh_level_t *L = mh_info->level;

	/* initialize entire memory hierarchy structure to benign values */
	for (i = 0; i < PAPI_MAX_MEM_HIERARCHY_LEVELS; i++) {
		for (j = 0; j < PAPI_MH_MAX_LEVELS; j++) {
			L[i].tlb[j].type = PAPI_MH_TYPE_EMPTY;
			L[i].tlb[j].num_entries = 0;
			L[i].tlb[j].associativity = 0;
			L[i].cache[j].type = PAPI_MH_TYPE_EMPTY;
			L[i].cache[j].size = 0;
			L[i].cache[j].line_size = 0;
			L[i].cache[j].num_lines = 0;
			L[i].cache[j].associativity = 0;
		}
	}
}

static short int _amd_L2_L3_assoc(unsigned short int pattern)
{
	/* From "CPUID Specification" #25481 Rev 2.28, April 2008 */
	short int assoc[16] = {0,1,2,-1,4,-1,8,-1,16,-1,32,48,64,96,128,SHRT_MAX};
	if (pattern > 0xF) return -1;
	return (assoc[pattern]);
}

/* Cache configuration for AMD AThlon/Duron */
static int init_amd(PAPI_mh_info_t * mh_info)
{
	union {
		struct {
			unsigned int ax, bx,cx,dx;
		} e;
		unsigned char byt[16];
	} reg;
	int i, j, levels = 0;
	PAPI_mh_level_t *L = mh_info->level;

   /*
	* Layout of CPU information taken from :
	* "CPUID Specification" #25481 Rev 2.28, April 2008 for most current info.
	*/

	MEMDBG("Initializing AMD memory info\n");
	/* AMD level 1 cache info */
	reg.e.ax = 0x80000005; /* extended function code 5: L1 Cache and TLB Identifiers */
	cpuid(&reg.e.ax, &reg.e.bx, &reg.e.cx, &reg.e.dx);

	MEMDBG("e.ax=0x%8.8x e.bx=0x%8.8x e.cx=0x%8.8x e.dx=0x%8.8x\n",
		reg.e.ax, reg.e.bx, reg.e.cx, reg.e.dx);
	MEMDBG(":\neax: %x %x %x %x\nebx: %x %x %x %x\necx: %x %x %x %x\nedx: %x %x %x %x\n",
		reg.byt[0],  reg.byt[1],  reg.byt[2],  reg.byt[3],
		reg.byt[4],  reg.byt[5],  reg.byt[6],  reg.byt[7],
		reg.byt[8],  reg.byt[9],  reg.byt[10], reg.byt[11],
		reg.byt[12], reg.byt[13], reg.byt[14], reg.byt[15]);

	/* NOTE: We assume L1 cache and TLB always exists */
	/* L1 TLB info */

	/* 4MB memory page information; half the number of entries as 2MB */
	L[0].tlb[0].type          = PAPI_MH_TYPE_INST;
	L[0].tlb[0].num_entries   = reg.byt[0]/2;
	L[0].tlb[0].page_size   = 4096 << 10;
	L[0].tlb[0].associativity = reg.byt[1];

	L[0].tlb[1].type          = PAPI_MH_TYPE_DATA;
	L[0].tlb[1].num_entries   = reg.byt[2]/2;
	L[0].tlb[1].page_size   = 4096 << 10;
	L[0].tlb[1].associativity = reg.byt[3];

	/* 2MB memory page information */
	L[0].tlb[2].type          = PAPI_MH_TYPE_INST;
	L[0].tlb[2].num_entries   = reg.byt[0];
	L[0].tlb[2].page_size   = 2048 << 10;
	L[0].tlb[2].associativity = reg.byt[1];

	L[0].tlb[3].type          = PAPI_MH_TYPE_DATA;
	L[0].tlb[3].num_entries   = reg.byt[2];
	L[0].tlb[3].page_size   = 2048 << 10;
	L[0].tlb[3].associativity = reg.byt[3];

	/* 4k page information */
	L[0].tlb[4].type          = PAPI_MH_TYPE_INST;
	L[0].tlb[4].num_entries   = reg.byt[4];
	L[0].tlb[4].page_size     = 4 << 10;
	L[0].tlb[4].associativity = reg.byt[5];

	L[0].tlb[5].type          = PAPI_MH_TYPE_DATA;
	L[0].tlb[5].num_entries   = reg.byt[6];
	L[0].tlb[5].page_size     = 4 << 10;
	L[0].tlb[5].associativity = reg.byt[7];

	for (i=0;i<PAPI_MH_MAX_LEVELS; i++) {
		if (L[0].tlb[i].associativity == 0xff)
			L[0].tlb[i].associativity = SHRT_MAX;
	}

	/* L1 D-cache info */
	L[0].cache[0].type = PAPI_MH_TYPE_DATA | PAPI_MH_TYPE_WB | PAPI_MH_TYPE_PSEUDO_LRU;
	L[0].cache[0].size = reg.byt[11]<<10;
	L[0].cache[0].associativity = reg.byt[10];
	L[0].cache[0].line_size = reg.byt[8];
	/* Byt[9] is "Lines per tag" */
	/* Is that == lines per cache? */
	/* L[0].cache[1].num_lines = reg.byt[9]; */
	if (L[0].cache[0].line_size)
		L[0].cache[0].num_lines = L[0].cache[0].size / L[0].cache[0].line_size;
	MEMDBG("D-Cache Line Count: %d; Computed: %d\n", reg.byt[9], L[0].cache[0].num_lines);

	/* L1 I-cache info */
	L[0].cache[1].type = PAPI_MH_TYPE_INST;
	L[0].cache[1].size = reg.byt[15]<<10;
	L[0].cache[1].associativity = reg.byt[14];
	L[0].cache[1].line_size = reg.byt[12];
	/* Byt[13] is "Lines per tag" */
	/* Is that == lines per cache? */
	/* L[0].cache[1].num_lines = reg.byt[13]; */
	if (L[0].cache[1].line_size)
		L[0].cache[1].num_lines = L[0].cache[1].size / L[0].cache[1].line_size;
	MEMDBG("I-Cache Line Count: %d; Computed: %d\n", reg.byt[13], L[0].cache[1].num_lines);

	for (i=0;i<2; i++) {
		if (L[0].cache[i].associativity == 0xff)
			L[0].cache[i].associativity = SHRT_MAX;
	}

	/* AMD L2/L3 Cache and L2 TLB info */
	/* NOTE: For safety we assume L2 and L3 cache and TLB may not exist */

	reg.e.ax = 0x80000006; /* extended function code 6: L2/L3 Cache and L2 TLB Identifiers */
	cpuid(&reg.e.ax, &reg.e.bx, &reg.e.cx, &reg.e.dx);

	MEMDBG("e.ax=0x%8.8x e.bx=0x%8.8x e.cx=0x%8.8x e.dx=0x%8.8x\n",
		reg.e.ax, reg.e.bx, reg.e.cx, reg.e.dx);
	MEMDBG(":\neax: %x %x %x %x\nebx: %x %x %x %x\necx: %x %x %x %x\nedx: %x %x %x %x\n",
		reg.byt[0],  reg.byt[1],  reg.byt[2],  reg.byt[3],
		reg.byt[4],  reg.byt[5],  reg.byt[6],  reg.byt[7],
		reg.byt[8],  reg.byt[9],  reg.byt[10], reg.byt[11],
		reg.byt[12], reg.byt[13], reg.byt[14], reg.byt[15]);

	/* L2 TLB info */

	if (reg.byt[0] | reg.byt[1]) { /* Level 2 ITLB exists */
		/* 4MB ITLB page information; half the number of entries as 2MB */
		L[1].tlb[0].type          = PAPI_MH_TYPE_INST;
		L[1].tlb[0].num_entries   = (((short)(reg.byt[1]&0xF)<<8) + reg.byt[0])/2;
		L[1].tlb[0].page_size     = 4096 << 10;
		L[1].tlb[0].associativity = _amd_L2_L3_assoc((reg.byt[1]&0xF0)>>4);

		/* 2MB ITLB page information */
		L[1].tlb[2].type          = PAPI_MH_TYPE_INST;
		L[1].tlb[2].num_entries   = L[1].tlb[0].num_entries * 2;
		L[1].tlb[2].page_size     = 2048 << 10;
		L[1].tlb[2].associativity = L[1].tlb[0].associativity;
	}

	if (reg.byt[2] | reg.byt[3]) { /* Level 2 DTLB exists */
		/* 4MB DTLB page information; half the number of entries as 2MB */
		L[1].tlb[1].type          = PAPI_MH_TYPE_DATA;
		L[1].tlb[1].num_entries   = (((short)(reg.byt[3]&0xF)<<8) + reg.byt[2])/2;
		L[1].tlb[1].page_size     = 4096 << 10;
		L[1].tlb[1].associativity = _amd_L2_L3_assoc((reg.byt[3]&0xF0)>>4);

		/* 2MB DTLB page information */
		L[1].tlb[3].type          = PAPI_MH_TYPE_DATA;
		L[1].tlb[3].num_entries   = L[1].tlb[1].num_entries * 2;
		L[1].tlb[3].page_size     = 2048 << 10;
		L[1].tlb[3].associativity = L[1].tlb[1].associativity;
	}

	/* 4k page information */
	if (reg.byt[4] | reg.byt[5]) { /* Level 2 ITLB exists */
		L[1].tlb[4].type          = PAPI_MH_TYPE_INST;
		L[1].tlb[4].num_entries   = ((short)(reg.byt[5]&0xF)<<8) + reg.byt[4];
		L[1].tlb[4].page_size     = 4 << 10;
		L[1].tlb[4].associativity = _amd_L2_L3_assoc((reg.byt[5]&0xF0)>>4);
	}
	if (reg.byt[6] | reg.byt[7]) { /* Level 2 DTLB exists */
		L[1].tlb[5].type          = PAPI_MH_TYPE_DATA;
		L[1].tlb[5].num_entries   = ((short)(reg.byt[7]&0xF)<<8) + reg.byt[6];
		L[1].tlb[5].page_size     = 4 << 10;
		L[1].tlb[5].associativity = _amd_L2_L3_assoc((reg.byt[7]&0xF0)>>4);
	}

	/* AMD Level 2 cache info */
	if (reg.e.cx) {
		L[1].cache[0].type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_WT | PAPI_MH_TYPE_PSEUDO_LRU;
		L[1].cache[0].size = ((reg.e.cx & 0xffff0000) >> 6); /* right shift by 16; multiply by 2^10 */
		L[1].cache[0].associativity = _amd_L2_L3_assoc((reg.byt[9]&0xF0)>>4);
		L[1].cache[0].line_size = reg.byt[8];
/*		L[1].cache[0].num_lines = reg.byt[9]&0xF; */
		if (L[1].cache[0].line_size)
			L[1].cache[0].num_lines = L[1].cache[0].size / L[1].cache[0].line_size;
		MEMDBG("U-Cache Line Count: %d; Computed: %d\n", reg.byt[9]&0xF, L[1].cache[0].num_lines);
	}

   /* AMD Level 3 cache info (shared across cores) */
	if (reg.e.dx) {
		L[2].cache[0].type = PAPI_MH_TYPE_UNIFIED | PAPI_MH_TYPE_WT | PAPI_MH_TYPE_PSEUDO_LRU;
		L[2].cache[0].size = reg.e.dx & 0xfffc0000; /* in blocks of 512KB (2^18) */
		L[2].cache[0].associativity = _amd_L2_L3_assoc((reg.byt[13]&0xF0)>>4);
		L[2].cache[0].line_size = reg.byt[12];
/*		L[2].cache[0].num_lines = reg.byt[13]&0xF; */
		if (L[2].cache[0].line_size)
			L[2].cache[0].num_lines = L[2].cache[0].size / L[2].cache[0].line_size;
		MEMDBG("U-Cache Line Count: %d; Computed: %d\n", reg.byt[13]&0xF, L[1].cache[0].num_lines);
	}
	for (i=0; i<PAPI_MAX_MEM_HIERARCHY_LEVELS; i++) {
		for (j=0; j<PAPI_MH_MAX_LEVELS; j++) {
			/* Compute the number of levels of hierarchy actually used */
			if (L[i].tlb[j].type != PAPI_MH_TYPE_EMPTY ||
				L[i].cache[j].type != PAPI_MH_TYPE_EMPTY)
				levels = i+1;
			}
		}
	return (levels);
}

   /*
	* "Intel® Processor Identification and the CPUID Instruction",
	* Application Note, AP-485, Nov 2008, 241618-033
	*
	* The following data structure and its instantiation trys to
	* capture all the information in Section 3.1.3 of the above
	* document. Not all of it is used by PAPI, but it could be.
	* As the above document is revised, this table should be
	* updated.
	*/

#define TLB_SIZES 3 /* number of different page sizes for a single TLB descriptor */
struct _intel_cache_info {
  int descriptor; /* 0x00 - 0xFF: register descriptor code */
  int level; /* 1 to PAPI_MH_MAX_LEVELS */
  int type;  /* Empty, instr, data, vector, unified | TLB */
  int size[TLB_SIZES];  /* cache or  TLB page size(s) in kB */
  int associativity; /* SHRT_MAX == fully associative */
  int sector; /* 1 if cache is sectored; else 0 */
  int line_size; /* for cache */
  int entries; /* for TLB */
};

static struct _intel_cache_info intel_cache[] = {
// 0x01
	{	.descriptor = 0x01,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_INST,
		.size[0] = 4,
		.associativity = 4,
		.entries = 32,
	},
// 0x02
	{	.descriptor = 0x02,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_INST,
		.size[0] = 4096,
		.associativity = SHRT_MAX,
		.entries = 2,
	},
// 0x03
	{	.descriptor = 0x03,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size[0] = 4,
		.associativity = 4,
		.entries = 64,
	},
// 0x04
	{	.descriptor = 0x04,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size[0] = 4096,
		.associativity = 4,
		.entries = 8,
	},
// 0x05
	{	.descriptor = 0x05,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size[0] = 4096,
		.associativity = 4,
		.entries = 32,
	},
// 0x06
	{	.descriptor = 0x06,
		.level = 1,
		.type = PAPI_MH_TYPE_INST,
		.size[0] = 8,
		.associativity = 4,
		.line_size = 32,
	},
// 0x08
	{	.descriptor = 0x08,
		.level = 1,
		.type = PAPI_MH_TYPE_INST,
		.size[0] = 16,
		.associativity = 4,
		.line_size = 32,
	},
// 0x09
	{	.descriptor = 0x09,
		.level = 1,
		.type = PAPI_MH_TYPE_INST,
		.size[0] = 32,
		.associativity = 4,
		.line_size = 64,
	},
// 0x0A
	{	.descriptor = 0x0A,
		.level = 1,
		.type = PAPI_MH_TYPE_DATA,
		.size[0] = 8,
		.associativity = 2,
		.line_size = 32,
	},
// 0x0C
	{	.descriptor = 0x0C,
		.level = 1,
		.type = PAPI_MH_TYPE_DATA,
		.size[0] = 16,
		.associativity = 4,
		.line_size = 32,
	},
// 0x0D
	{	.descriptor = 0x0D,
		.level = 1,
		.type = PAPI_MH_TYPE_DATA,
		.size[0] = 16,
		.associativity = 4,
		.line_size = 64,
	},
// 0x21
	{	.descriptor = 0x21,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 256,
		.associativity = 8,
		.line_size = 64,
	},
// 0x22
	{	.descriptor = 0x22,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 512,
		.associativity = 4,
		.sector = 1,
		.line_size = 64,
	},
// 0x23
	{	.descriptor = 0x23,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 1024,
		.associativity = 8,
		.sector = 1,
		.line_size = 64,
	},
// 0x25
	{	.descriptor = 0x25,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 2048,
		.associativity = 8,
		.sector = 1,
		.line_size = 64,
	},
// 0x29
	{	.descriptor = 0x29,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 4096,
		.associativity = 8,
		.sector = 1,
		.line_size = 64,
	},
// 0x2C
	{	.descriptor = 0x2C,
		.level = 1,
		.type = PAPI_MH_TYPE_DATA,
		.size[0] = 32,
		.associativity = 8,
		.line_size = 64,
	},
// 0x30
	{	.descriptor = 0x30,
		.level = 1,
		.type = PAPI_MH_TYPE_INST,
		.size[0] = 32,
		.associativity = 8,
		.line_size = 64,
	},
// 0x39
	{	.descriptor = 0x39,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 128,
		.associativity = 4,
		.sector = 1,
		.line_size = 64,
	},
// 0x3A
	{	.descriptor = 0x3A,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 192,
		.associativity = 6,
		.sector = 1,
		.line_size = 64,
	},
// 0x3B
	{	.descriptor = 0x3B,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 128,
		.associativity = 2,
		.sector = 1,
		.line_size = 64,
	},
// 0x3C
	{	.descriptor = 0x3C,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 256,
		.associativity = 4,
		.sector = 1,
		.line_size = 64,
	},
// 0x3D
	{	.descriptor = 0x3D,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 384,
		.associativity = 6,
		.sector = 1,
		.line_size = 64,
	},
// 0x3E
	{	.descriptor = 0x3E,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 512,
		.associativity = 4,
		.sector = 1,
		.line_size = 64,
	},
// 0x40: no last level cache (??)
// 0x41
	{	.descriptor = 0x41,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 128,
		.associativity = 4,
		.line_size = 32,
	},
// 0x42
	{	.descriptor = 0x42,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 256,
		.associativity = 4,
		.line_size = 32,
	},
// 0x43
	{	.descriptor = 0x43,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 512,
		.associativity = 4,
		.line_size = 32,
	},
// 0x44
	{	.descriptor = 0x44,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 1024,
		.associativity = 4,
		.line_size = 32,
	},
// 0x45
	{	.descriptor = 0x45,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 2048,
		.associativity = 4,
		.line_size = 32,
	},
// 0x46
	{	.descriptor = 0x46,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 4096,
		.associativity = 4,
		.line_size = 64,
	},
// 0x47
	{	.descriptor = 0x47,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 8192,
		.associativity = 8,
		.line_size = 64,
	},
// 0x48
	{	.descriptor = 0x48,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 3072,
		.associativity = 12,
		.line_size = 64,
	},
// 0x49 NOTE: for family 0x0F model 0x06 this is level 3
	{	.descriptor = 0x49,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 4096,
		.associativity = 16,
		.line_size = 64,
	},
// 0x4A
	{	.descriptor = 0x4A,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 6144,
		.associativity = 12,
		.line_size = 64,
	},
// 0x4B
	{	.descriptor = 0x4B,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 8192,
		.associativity = 16,
		.line_size = 64,
	},
// 0x4C
	{	.descriptor = 0x4C,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 12288,
		.associativity = 12,
		.line_size = 64,
	},
// 0x4D
	{	.descriptor = 0x4D,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 16384,
		.associativity = 16,
		.line_size = 64,
	},
// 0x4E
	{	.descriptor = 0x4E,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 6144,
		.associativity = 24,
		.line_size = 64,
	},
// 0x50
	{	.descriptor = 0x50,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_INST,
		.size = {4, 2048, 4096},
		.associativity = SHRT_MAX,
		.entries = 64,
	},
// 0x51
	{	.descriptor = 0x51,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_INST,
		.size = {4, 2048, 4096},
		.associativity = SHRT_MAX,
		.entries = 128,
	},
// 0x52
	{	.descriptor = 0x52,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_INST,
		.size = {4, 2048, 4096},
		.associativity = SHRT_MAX,
		.entries = 256,
	},
// 0x55
	{	.descriptor = 0x55,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_INST,
		.size = {2048, 4096, 0},
		.associativity = SHRT_MAX,
		.entries = 7,
	},
// 0x56
	{	.descriptor = 0x56,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size[0] = 4096,
		.associativity = 4,
		.entries = 16,
	},
// 0x57
	{	.descriptor = 0x57,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size[0] = 4,
		.associativity = 4,
		.entries = 16,
	},
// 0x5A
	{	.descriptor = 0x5A,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size = {2048, 4096, 0},
		.associativity = 4,
		.entries = 32,
	},
// 0x5B
	{	.descriptor = 0x5B,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size = {4, 4096, 0},
		.associativity = SHRT_MAX,
		.entries = 64,
	},
// 0x5C
	{	.descriptor = 0x5C,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size = {4, 4096, 0},
		.associativity = SHRT_MAX,
		.entries = 128,
	},
// 0x5D
	{	.descriptor = 0x5D,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size = {4, 4096, 0},
		.associativity = SHRT_MAX,
		.entries = 256,
	},
// 0x60
	{	.descriptor = 0x60,
		.level = 1,
		.type = PAPI_MH_TYPE_DATA,
		.size[0] = 16,
		.associativity = 8,
		.sector = 1,
		.line_size = 64,
	},
// 0x66
	{	.descriptor = 0x66,
		.level = 1,
		.type = PAPI_MH_TYPE_DATA,
		.size[0] = 8,
		.associativity = 4,
		.sector = 1,
		.line_size = 64,
	},
// 0x67
	{	.descriptor = 0x67,
		.level = 1,
		.type = PAPI_MH_TYPE_DATA,
		.size[0] = 16,
		.associativity = 4,
		.sector = 1,
		.line_size = 64,
	},
// 0x68
	{	.descriptor = 0x68,
		.level = 1,
		.type = PAPI_MH_TYPE_DATA,
		.size[0] = 32,
		.associativity = 4,
		.sector = 1,
		.line_size = 64,
	},
// 0x70
	{	.descriptor = 0x70,
		.level = 1,
		.type = PAPI_MH_TYPE_TRACE,
		.size[0] = 12,
		.associativity = 8,
	},
// 0x71
	{	.descriptor = 0x71,
		.level = 1,
		.type = PAPI_MH_TYPE_TRACE,
		.size[0] = 16,
		.associativity = 8,
	},
// 0x72
	{	.descriptor = 0x72,
		.level = 1,
		.type = PAPI_MH_TYPE_TRACE,
		.size[0] = 32,
		.associativity = 8,
	},
// 0x73
	{	.descriptor = 0x73,
		.level = 1,
		.type = PAPI_MH_TYPE_TRACE,
		.size[0] = 64,
		.associativity = 8,
	},
// 0x78
	{	.descriptor = 0x78,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 1024,
		.associativity = 4,
		.line_size = 64,
	},
// 0x79
	{	.descriptor = 0x79,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 128,
		.associativity = 8,
		.sector = 1,
		.line_size = 64,
	},
// 0x7A
	{	.descriptor = 0x7A,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 256,
		.associativity = 8,
		.sector = 1,
		.line_size = 64,
	},
// 0x7B
	{	.descriptor = 0x7B,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 512,
		.associativity = 8,
		.sector = 1,
		.line_size = 64,
	},
// 0x7C
	{	.descriptor = 0x7C,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 1024,
		.associativity = 8,
		.sector = 1,
		.line_size = 64,
	},
// 0x7D
	{	.descriptor = 0x7D,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 2048,
		.associativity = 8,
		.line_size = 64,
	},
// 0x7F
	{	.descriptor = 0x7F,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 512,
		.associativity = 2,
		.line_size = 64,
	},
// 0x82
	{	.descriptor = 0x82,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 256,
		.associativity = 8,
		.line_size = 32,
	},
// 0x83
	{	.descriptor = 0x83,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 512,
		.associativity = 8,
		.line_size = 32,
	},
// 0x84
	{	.descriptor = 0x84,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 1024,
		.associativity = 8,
		.line_size = 32,
	},
// 0x85
	{	.descriptor = 0x85,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 2048,
		.associativity = 8,
		.line_size = 32,
	},
// 0x86
	{	.descriptor = 0x86,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 512,
		.associativity = 4,
		.line_size = 64,
	},
// 0x87
	{	.descriptor = 0x87,
		.level = 2,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 1024,
		.associativity = 8,
		.line_size = 64,
	},
// 0xB0
	{	.descriptor = 0xB0,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_INST,
		.size[0] = 4,
		.associativity = 4,
		.entries = 128,
	},
// 0xB1 NOTE: This is currently the only instance where .entries
//		is dependent on .size. It's handled as a code exception.
//		If other instances appear in the future, the structure
//		should probably change to accomodate it.
	{	.descriptor = 0xB1,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_INST,
		.size = {2048, 4096, 0},
		.associativity = 4,
		.entries = 8, /* or 4 if size = 4096 */
	},
// 0xB2
	{	.descriptor = 0xB2,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_INST,
		.size[0] = 4,
		.associativity = 4,
		.entries = 64,
	},
// 0xB3
	{	.descriptor = 0xB3,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size[0] = 4,
		.associativity = 4,
		.entries = 128,
	},
// 0xB4
	{	.descriptor = 0xB4,
		.level = 1,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_DATA,
		.size[0] = 4,
		.associativity = 4,
		.entries = 256,
	},
// 0xCA
	{	.descriptor = 0xCA,
		.level = 2,
		.type = PAPI_MH_TYPE_TLB | PAPI_MH_TYPE_UNIFIED,
		.size[0] = 4,
		.associativity = 4,
		.entries = 512,
	},
// 0xD0
	{	.descriptor = 0xD0,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 512,
		.associativity = 4,
		.line_size = 64,
	},
// 0xD1
	{	.descriptor = 0xD1,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 1024,
		.associativity = 4,
		.line_size = 64,
	},
// 0xD2
	{	.descriptor = 0xD2,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 2048,
		.associativity = 4,
		.line_size = 64,
	},
// 0xD6
	{	.descriptor = 0xD6,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 1024,
		.associativity = 8,
		.line_size = 64,
	},
// 0xD7
	{	.descriptor = 0xD7,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 2048,
		.associativity = 8,
		.line_size = 64,
	},
// 0xD8
	{	.descriptor = 0xD8,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 4096,
		.associativity = 8,
		.line_size = 64,
	},
// 0xDC
	{	.descriptor = 0xDC,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 2048,
		.associativity = 12,
		.line_size = 64,
	},
// 0xDD
	{	.descriptor = 0xDD,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 4096,
		.associativity = 12,
		.line_size = 64,
	},
// 0xDE
	{	.descriptor = 0xDE,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 8192,
		.associativity = 12,
		.line_size = 64,
	},
// 0xE2
	{	.descriptor = 0xE2,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 2048,
		.associativity = 16,
		.line_size = 64,
	},
// 0xE3
	{	.descriptor = 0xE3,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 4096,
		.associativity = 16,
		.line_size = 64,
	},
// 0xE4
	{	.descriptor = 0xE4,
		.level = 3,
		.type = PAPI_MH_TYPE_UNIFIED,
		.size[0] = 8192,
		.associativity = 16,
		.line_size = 64,
	},
// 0xF0
	{	.descriptor = 0xF0,
		.level = 1,
		.type = PAPI_MH_TYPE_PREF,
		.size[0] = 64,
	},
// 0xF1
	{	.descriptor = 0xF1,
		.level = 1,
		.type = PAPI_MH_TYPE_PREF,
		.size[0] = 128,
	},
};

#ifdef DEBUG
static void print_intel_cache_table() {
	int i,j;
	for (i=0;i<(sizeof(intel_cache)/sizeof(struct _intel_cache_info));i++) {
		printf("%d.\tDescriptor: 0x%x\n", i, intel_cache[i].descriptor);
		printf("\t  Level:     %d\n", intel_cache[i].level);
		printf("\t  Type:      %d\n", intel_cache[i].type);
		printf("\t  Size(s):   ");
		for (j=0; j<TLB_SIZES; j++)
			printf("%d, ", intel_cache[i].size[j]);
		printf("\n");
		printf("\t  Assoc:     %d\n", intel_cache[i].associativity);
		printf("\t  Sector:    %d\n", intel_cache[i].sector);
		printf("\t  Line Size: %d\n", intel_cache[i].line_size);
		printf("\t  Entries:   %d\n", intel_cache[i].entries);
		printf("\n");
	}
}
#endif

/* Given a specific cache descriptor, this routine decodes the information from a table
 * of such descriptors and fills out one or more records in a PAPI data structure.
 * Called only by init_intel()
 */
static void intel_decode_descriptor(struct _intel_cache_info *d, PAPI_mh_level_t *L) {
	int i, next;
	int level = d->level - 1;
	PAPI_mh_tlb_info_t *t;
	PAPI_mh_cache_info_t *c;

	if (d->descriptor == 0x49) { /* special case */
		unsigned int r_eax, r_ebx, r_ecx, r_edx;
		r_eax = 0x1; /* function code 1: family & model */
		cpuid(&r_eax, &r_ebx, &r_ecx, &r_edx);
		/* override table for Family F, model 6 only */
		if ((r_eax & 0x0FFF3FF0) == 0xF60) level = 3;
	}
	if (d->type & PAPI_MH_TYPE_TLB) {
		for (next = 0; next < PAPI_MH_MAX_LEVELS-1; next++) {
			if (L[level].tlb[next].type == PAPI_MH_TYPE_EMPTY) break;
		}
		/* expand TLB entries for multiple possible page sizes */
		for (i=0; i<TLB_SIZES && next<PAPI_MH_MAX_LEVELS && d->size[i]; i++, next++) {
//			printf("Level %d Descriptor: %x TLB type %x next: %d, i: %d\n", level, d->descriptor, d->type, next, i);
			t = &L[level].tlb[next];
			t->type = PAPI_MH_CACHE_TYPE(d->type);
			t->num_entries = d->entries;
			t->page_size = d->size[i] << 10; /* minimum page size in KB*/
			t->associativity = d->associativity;
			/* another special case */
			if (d->descriptor == 0xB1 && d->size[i] == 4096)
				t->num_entries = d->entries/2;
		}
	} else {
		for (next = 0; next < PAPI_MH_MAX_LEVELS-1; next++) {
			if (L[level].cache[next].type == PAPI_MH_TYPE_EMPTY) break;
		}
//		printf("Level %d Descriptor: %x Cache type %x next: %d\n", level, d->descriptor, d->type, next);
		c = &L[level].cache[next];
		c->type = PAPI_MH_CACHE_TYPE(d->type);
		c->size = d->size[0] << 10; /* convert from KB to bytes */
		c->associativity = d->associativity;
		if (d->line_size) {
			c->line_size = d->line_size;
			c->num_lines = c->size / c->line_size;
		}
	}
}

static int init_intel(PAPI_mh_info_t * mh_info)
{
	/* cpuid() returns memory copies of 4 32-bit registers
	 * this union allows them to be accessed as either registers
	 * or individual bytes. Remember that Intel is little-endian.
	 */
	union {
		struct {
			unsigned int ax, bx,cx,dx;
		} e;
		unsigned char descrip[16];
	} reg;

	int r; /* register boundary index */
	int b; /* byte index into a register */
	int i; /* byte index into the descrip array */
	int t; /* table index into the static descriptor table */
	int count; /* how many times to call cpuid; from eax:lsb */
	int size;  /* size of the descriptor table */
	int last_level = 0; /* how many levels in the cache hierarchy */

	/* All of Intel's cache info is in 1 call to cpuid
	 * however it is a table lookup :(
	*/
	MEMDBG("Initializing Intel Cache and TLB descriptors\n");

#ifdef DEBUG
	if (ISLEVEL(DEBUG_MEMORY))
		print_intel_cache_table();
#endif

	reg.e.ax = 0x2; /* function code 2: cache descriptors */
	cpuid(&reg.e.ax, &reg.e.bx, &reg.e.cx, &reg.e.dx);

	MEMDBG("e.ax=0x%8.8x e.bx=0x%8.8x e.cx=0x%8.8x e.dx=0x%8.8x\n",
		reg.e.ax, reg.e.bx, reg.e.cx, reg.e.dx);
	MEMDBG(":\nd0: %x %x %x %x\nd1: %x %x %x %x\nd2: %x %x %x %x\nd3: %x %x %x %x\n",
		reg.descrip[0], reg.descrip[1], reg.descrip[2], reg.descrip[3],
		reg.descrip[4], reg.descrip[5], reg.descrip[6], reg.descrip[7],
		reg.descrip[8], reg.descrip[9], reg.descrip[10], reg.descrip[11],
		reg.descrip[12], reg.descrip[13], reg.descrip[14], reg.descrip[15]);

	count = reg.descrip[0]; /* # times to repeat CPUID call. Not implemented. */
	size = (sizeof(intel_cache)/sizeof(struct _intel_cache_info)); /* # descriptors */
	MEMDBG("Repeat cpuid(2,...) %d times. If not 1, code is broken.\n", count);
	for (r = 0; r < 4; r++) { /* walk the registers */
		if ((reg.descrip[r*4+3] & 0x80) == 0) { /* only process if high order bit is 0 */
			for (b = 3; b >= 0; b--) { /* walk the descriptor bytes from high to low */
				i = r*4+b; /* calculate an index into the array of descriptors */
				if (i) { /* skip the low order byte in eax [0]; it's the count (see above) */
					for (t = 0; t < size; t++) { /* walk the descriptor table */
						if (reg.descrip[i] == intel_cache[t].descriptor) { /* find match */
							if (intel_cache[t].level > last_level)
								last_level = intel_cache[t].level;
							intel_decode_descriptor(&intel_cache[t], mh_info->level);
						}
					}
				}
			}
		}
	}
	MEMDBG("# of Levels: %d\n",last_level);
	return(last_level);
}

/* Checks to see if cpuid exists on this processor, if
 * it doesn't it is pre pentium K6 series that we don't
 * support.
 */
#if 0
/* This routine appears to no longer be called. */
static int check_cpuid()
{
   volatile unsigned long val;
#ifdef _WIN32
   __asm {
	   pushfd
	   pop eax
	   mov ebx, eax
	   xor eax, 0x00200000
	   push eax 
	   popfd 
	   pushfd 
	   pop eax 
	   cmp eax, ebx 
	   jz NO_CPUID 
	   mov val, 1 
	   jmp END 
NO_CPUID:  mov val, 0 
END:	  }
#elif defined(__x86_64__)
   __asm__ __volatile__("pushf;"
                        "pop %%eax;"
                        "mov %%eax, %%ebx;"
                        "xor $0x00200000,%%eax;"
                        "push %%eax;"
                        "popf;"
                        "pushf;"
                        "pop %%eax;"
                        "cmp %%eax, %%ebx;"
                        "jz NO_CPUID;"
                        "mov $1, %0;"
                        "jmp END;" "NO_CPUID:" "mov $0, %0;" "END:":"=r"(val));
#else
   __asm__ __volatile__("pushfl;"
                        "pop %%eax;"
                        "movl %%eax, %%ebx;"
                        "xor $0x00200000,%%eax;"
                        "push %%eax;"
                        "popfl;"
                        "pop %%eax;"
                        "cmp %%eax, %%ebx;"
                        "jz NO_CPUID;"
                        "movl $1, %0;"
                        "jmp END;" "NO_CPUID:" "movl $0, %0;" "END:":"=r"(val));
#endif
   return (int) val;
}
#endif
#ifdef _WIN32
static void cpuid(unsigned int *a, unsigned int *b,
                         unsigned int *c, unsigned int *d)
{
   volatile unsigned long tmp, tmp2, tmp3, tmp4;
   volatile unsigned long in_tmp;

   in_tmp = *a;
   __asm {
      mov eax, in_tmp;
      cpuid;
      mov tmp, eax;
      mov tmp2, ebx;
      mov tmp3, ecx;
      mov tmp4, edx;
   }
   *a = tmp;
   *b = tmp2;
   *c = tmp3;
   *d = tmp4;
}
#else
inline_static void cpuid(unsigned int *a, unsigned int *b,
                  unsigned int *c, unsigned int *d)
{
  unsigned int op = *a;
  // .byte 0x53 == push ebx. it's universal for 32 and 64 bit
  // .byte 0x5b == pop ebx.
  // Some gcc's (4.1.2 on Core2) object to pairing push/pop and ebx in 64 bit mode.
  // Using the opcode directly avoids this problem.
  __asm__ __volatile__ (".byte 0x53\n\tcpuid\n\tmovl %%ebx, %%esi\n\t.byte 0x5b"
       : "=a" (*a),
	     "=S" (*b),
		 "=c" (*c),
		 "=d" (*d)
       : "a" (op));
}
#endif

/* A pointer to the following is passed to PAPI_get_dmem_info() 
	typedef struct _dmem_t {
	  long long size;
	  long long resident;
	  long long high_water_mark;
	  long long shared;
	  long long text;
	  long long library;
	  long long heap;
	  long long locked;
	  long long stack;
	  long long pagesize;
	} PAPI_dmem_info_t;
*/


#ifdef _WIN32
#include <Psapi.h>
int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *d)
{

   HANDLE proc = GetCurrentProcess();
   PROCESS_MEMORY_COUNTERS cntr;
   SYSTEM_INFO SystemInfo;      // system information structure  

   GetSystemInfo(&SystemInfo);
   GetProcessMemoryInfo(proc, &cntr, sizeof(cntr));

   d->pagesize = SystemInfo.dwPageSize;
   d->size = (cntr.WorkingSetSize - cntr.PagefileUsage) / SystemInfo.dwPageSize;
   d->resident = cntr.WorkingSetSize / SystemInfo.dwPageSize;
   d->high_water_mark = cntr.PeakWorkingSetSize / SystemInfo.dwPageSize;
  
   return PAPI_OK;
}

#else
#ifdef __CATAMOUNT__
int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *d)
{
	return( PAPI_EINVAL );
}
#else
//int get_dmem_info(long long *size, long long *resident, long long *shared, long long *text, long long *library, long long *heap, long long *locked, long long *stack, long long *ps, long long *vmhwm)
int _papi_hwd_get_dmem_info(PAPI_dmem_info_t *d)
{
  char fn[PATH_MAX], tmp[PATH_MAX];
  FILE *f;
  int ret;
  long long sz = 0, lck = 0, res = 0, shr = 0, stk = 0, txt = 0, dat = 0, dum = 0, lib = 0, hwm = 0;

  sprintf(fn,"/proc/%ld/status",(long)getpid());
  f = fopen(fn,"r");
  if (f == NULL)
    {
      PAPIERROR("fopen(%s): %s\n",fn,strerror(errno));
      return PAPI_ESBSTR;
    }
  while (1)
    {
      if (fgets(tmp,PATH_MAX,f) == NULL)
	break;
      if (strspn(tmp,"VmSize:") == strlen("VmSize:"))
	{
	  sscanf(tmp+strlen("VmSize:"),"%lld",&sz);
	  d->size = sz;
	  continue;
	}
      if (strspn(tmp,"VmHWM:") == strlen("VmHWM:"))
	{
	  sscanf(tmp+strlen("VmHWM:"),"%lld",&hwm);
	  d->high_water_mark = hwm;
	  continue;
	}
      if (strspn(tmp,"VmLck:") == strlen("VmLck:"))
	{
	  sscanf(tmp+strlen("VmLck:"),"%lld",&lck);
	  d->locked = lck;
	  continue;
	}
      if (strspn(tmp,"VmRSS:") == strlen("VmRSS:"))
	{
	  sscanf(tmp+strlen("VmRSS:"),"%lld",&res);
	  d->resident = res;
	  continue;
	}
      if (strspn(tmp,"VmData:") == strlen("VmData:"))
	{
	  sscanf(tmp+strlen("VmData:"),"%lld",&dat);
	  d->heap = dat;
	  continue;
	}
      if (strspn(tmp,"VmStk:") == strlen("VmStk:"))
	{
	  sscanf(tmp+strlen("VmStk:"),"%lld",&stk);
	  d->stack = stk;
	  continue;
	}
      if (strspn(tmp,"VmExe:") == strlen("VmExe:"))
	{
	  sscanf(tmp+strlen("VmExe:"),"%lld",&txt);
	  d->text = txt;
	  continue;
	}
      if (strspn(tmp,"VmLib:") == strlen("VmLib:"))
	{
	  sscanf(tmp+strlen("VmLib:"),"%lld",&lib);
	  d->library = lib;
	  continue;
	}
    }
  fclose(f);

  sprintf(fn,"/proc/%ld/statm",(long)getpid());
  f = fopen(fn,"r");
  if (f == NULL)
    {
      PAPIERROR("fopen(%s): %s\n",fn,strerror(errno));
      return PAPI_ESBSTR;
    }
  ret = fscanf(f,"%lld %lld %lld %lld %lld %lld %lld",&dum,&dum,&shr,&dum,&dum,&dat,&dum);
  if (ret != 7)
    {
      PAPIERROR("fscanf(7 items): %d\n",ret);
      return PAPI_ESBSTR;
    }
  d->pagesize = getpagesize();
  d->shared = (shr * d->pagesize)/1024;
  fclose(f);

  return PAPI_OK;
}

#endif /* __CATAMOUNT__ */
#endif /* _WIN32 */


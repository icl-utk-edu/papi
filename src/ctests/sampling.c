/*
* File:    sampling.c
* Author:  Vince Weaver <vincent.weaver@maine.edu>
*/

/* This file demonstrates the new sampling interface.
   It writes the raw sampling info to a file on disk.
*/

#include <stdio.h>
#include <stdlib.h>

#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <fcntl.h>

#include <errno.h>

#include <signal.h>

#include <sys/mman.h>

#include <sys/ioctl.h>
#include <asm/unistd.h>
#include <sys/prctl.h>

#include "./perf_event.h"

//#include "perf_helpers.h"
//#include "perf_barrier.h"

#define RAW_NONE        0
#define RAW_IBS_FETCH   1
#define RAW_IBS_OP      2

#define MMAP_DATA_SIZE	1

static int sample_type;
static int global_sample_regs_user;
static int quiet;

#ifdef __powerpc__
#define rmb() asm volatile ("sync" : : : "memory")

#elif defined (__s390__)
#define rmb() asm volatile("bcr 15,0" ::: "memory")

#elif defined (__sh__)
#if defined(__SH4A__) || defined(__SH5__)
#define rmb()          asm volatile("synco" ::: "memory")
#else
#define rmb()          asm volatile("" ::: "memory")
#endif

#elif defined (__hppa__)
#define rmb()           asm volatile("" ::: "memory")

#elif defined (__sparc__)
#define rmb()           asm volatile("":::"memory")

#elif defined (__alpha__)
#define rmb()           asm volatile("mb" ::: "memory")

#elif defined(__ia64__)
#define rmb()           asm volatile ("mf" ::: "memory")

#elif defined(__arm__)
/*
 * Use the __kuser_memory_barrier helper in the CPU helper page. See
 * arch/arm/kernel/entry-armv.S in the kernel source for details.
 */
#define rmb()           ((void(*)(void))0xffff0fa0)()

//#if __LINUX_ARM_ARCH__ >= 7
//#define rmb() asm volatile("dsb " "" : : : "memory")
//#else
//#define rmb() asm volatile("mcr p15, 0, %0, c7, c10, 4" : : "r" (0) : "memory")
//#endif

#elif defined(__aarch64__)
#define rmb()           asm volatile("dmb ld" ::: "memory")
//#define rmb() asm volatile("dsb " "ld" : : : "memory")

#elif defined(__mips__)
#define rmb()           asm volatile(                                   \
                                ".set   mips2\n\t"                      \
                                "sync\n\t"                              \
                                ".set   mips0"                          \
				: /* no output */                       \
				: /* no input */                        \
				: "memory")

#elif defined(__i386__)
#define rmb() asm volatile("lock; addl $0,0(%%esp)" ::: "memory")

#elif defined(__x86_64)

#if defined(__KNC__)
#define rmb() __sync_synchronize()

#else
#define rmb() asm volatile("lfence":::"memory")
#endif

#else
#error Need to define rmb for this architecture!
#error See the kernel source directory: tools/perf/perf.h file
#endif

//#if defined (__x86_64)
//#define mb() 	asm volatile("mfence":::"memory")
//#endif


//#include "parse_record.h"

#if defined(__x86_64__) || defined(__i386__) ||defined(__arm__)
#include <asm/perf_regs.h>
#endif

#include "papi.h"
#include "papi_test.h"

#include "do_loops.h"

static int total = 0;				   /* total overflows */

static long long prev_head=0;

/* parse_record.c  */
/* by Vince Weaver   vincent.weaver _at_ maine.edu */

/* This code parses the records stored in a perf mmap trace buffer */



/* Urgh who designed this interface */
static int handle_struct_read_format(unsigned char *sample,
				     int read_format,
				     int quiet) {

	int offset=0,i;

	if (read_format & PERF_FORMAT_GROUP) {
		long long nr,time_enabled,time_running;

		memcpy(&nr,&sample[offset],sizeof(long long));
		if (!quiet) printf("\t\tNumber: %lld ",nr);
		offset+=8;

		if (read_format & PERF_FORMAT_TOTAL_TIME_ENABLED) {
			memcpy(&time_enabled,&sample[offset],sizeof(long long));
			if (!quiet) printf("enabled: %lld ",time_enabled);
			offset+=8;
		}
		if (read_format & PERF_FORMAT_TOTAL_TIME_RUNNING) {
			memcpy(&time_running,&sample[offset],sizeof(long long));
			if (!quiet) printf("running: %lld ",time_running);
			offset+=8;
		}

		if (!quiet) printf("\n");

		for(i=0;i<nr;i++) {
			long long value, id;

			memcpy(&value,&sample[offset],sizeof(long long));
			if (!quiet) printf("\t\t\tValue: %lld ",value);
			offset+=8;

			if (read_format & PERF_FORMAT_ID) {
				memcpy(&id,&sample[offset],sizeof(long long));
				if (!quiet) printf("id: %lld ",id);
				offset+=8;
			}

			if (!quiet) printf("\n");
		}
	}
	else {

		long long value,time_enabled,time_running,id;

		memcpy(&value,&sample[offset],sizeof(long long));
		if (!quiet) printf("\t\tValue: %lld ",value);
		offset+=8;

		if (read_format & PERF_FORMAT_TOTAL_TIME_ENABLED) {
			memcpy(&time_enabled,&sample[offset],sizeof(long long));
			if (!quiet) printf("enabled: %lld ",time_enabled);
			offset+=8;
		}
		if (read_format & PERF_FORMAT_TOTAL_TIME_RUNNING) {
			memcpy(&time_running,&sample[offset],sizeof(long long));
			if (!quiet) printf("running: %lld ",time_running);
			offset+=8;
		}
		if (read_format & PERF_FORMAT_ID) {
			memcpy(&id,&sample[offset],sizeof(long long));
			if (!quiet) printf("id: %lld ",id);
			offset+=8;
		}
		if (!quiet) printf("\n");
	}

	return offset;
}

#if defined(__x86_64__)

#define NUM_REGS	PERF_REG_X86_64_MAX
static char reg_names[NUM_REGS][8]=
			{"RAX","RBX","RCX","RDX","RSI","RDI","RBP","RSP",
			 "RIP","RFLAGS","CS","SS","DS","ES","FS","GS",
			 "R8","R9","R10","R11","R12","R13","R14","R15"};


#elif defined(__i386__)

#define NUM_REGS	PERF_REG_X86_32_MAX
static char reg_names[PERF_REG_X86_32_MAX][8]=
			{"EAX","EBX","ECX","EDX","ESI","EDI","EBP","ESP",
			 "EIP","EFLAGS","CS","SS","DS","ES","FS","GS"};

#elif defined(__arm__)

#define NUM_REGS	PERF_REG_ARM_MAX
static char reg_names[PERF_REG_ARM_MAX][8]=
			{"R0","R1","R2","R3","R4","R5","R6","R7",
			 "R8","R9","R10","FP","IP","SP","LR","PC"};

#else

#define NUM_REGS 0

static char reg_names[1][8]={"NONE!"};

#endif



static int print_regs(int quiet,long long abi,long long reg_mask,
		unsigned char *data) {

	int return_offset=0;
	int num_regs=NUM_REGS;
	int i;
	unsigned long long reg_value;

	if (!quiet) printf("\t\tReg mask %llx\n",reg_mask);
	for(i=0;i<64;i++) {
		if (reg_mask&1ULL<<i) {
			if (!quiet) {
				memcpy(&reg_value,&data[return_offset],8);
				if (i<num_regs) {
					printf("\t\t%s : ",reg_names[i]);
				}
				else {
					printf("\t\t??? : ");
				}

				printf("%llx\n",reg_value);
			}
			return_offset+=8;
		}
	}

	return return_offset;
}


static int dump_raw_ibs_fetch(unsigned char *data, int size) {

	unsigned long long *msrs;
	unsigned int *leftover;

	msrs=(unsigned long long *)(data+4);
	leftover=(unsigned int *)(data);

	printf("\t\tHeader: %x\n",leftover[0]);
	printf("\t\tMSR IBS_FETCH_CONTROL %llx\n",msrs[0]);
	printf("\t\t\tIBS_RAND_EN: %d\n",!!(msrs[0]&1ULL<<57));
	printf("\t\t\tL2 iTLB Miss: %d\n",!!(msrs[0]&1ULL<<56));
	printf("\t\t\tL1 iTLB Miss: %d\n",!!(msrs[0]&1ULL<<55));
	printf("\t\t\tL1TLB page size: ");
	switch( (msrs[0]>>53)&0x3) {
		case 0:	printf("4kB\n"); break;
		case 1:	printf("2MB\n"); break;
		case 2: printf("1GB\n"); break;
		default:	printf("Resreved\n"); break;
	}

	printf("\t\t\tFetch Physical Address Valid: %d\n",!!(msrs[0]&1ULL<<52));
	printf("\t\t\ticache miss: %d\n",!!(msrs[0]&1ULL<<51));
	printf("\t\t\tInstruction Fetch Complete: %d\n",!!(msrs[0]&1ULL<<50));
	printf("\t\t\tInstruction Fetch Valid: %d\n",!!(msrs[0]&1ULL<<49));
	printf("\t\t\tInstruction Fetch Enabled: %d\n",!!(msrs[0]&1ULL<<48));
	printf("\t\t\tInstruction Fetch Latency: %lld\n",((msrs[0]>>32)&0xffff));
	printf("\t\t\tInstruction Fetch Count: %lld\n",((msrs[0]>>16)&0xffff)<<4);
	printf("\t\t\tInstruction Fetch Max Count: %lld\n",(msrs[0]&0xffff)<<4);

	printf("\t\tMSR IBS_FETCH_LINEAR_ADDRESS %llx\n",msrs[1]);
	printf("\t\tMSR IBS_FETCH_PHYSICAL_ADDRESS %llx\n",msrs[2]);
	if (size>24) {
		printf("\t\tMSR IBS_BRTARGET %llx\n",msrs[3]);
	}
	return 0;
}

static int dump_raw_ibs_op(unsigned char *data, int size) {

	unsigned long long *msrs;
	unsigned int *leftover;

	msrs=(unsigned long long *)(data+4);
	leftover=(unsigned int *)(data);

	printf("\t\tHeader: %x\n",leftover[0]);
	printf("\t\tMSR IBS_EXECUTION_CONTROL %llx\n",msrs[0]);
	printf("\t\t\tIbsOpCurCnt: %lld\n",((msrs[0]>>32)&0x3ffffff));
	printf("\t\t\tIBS OpCntCtl: %d\n",!!(msrs[0]&1ULL<<19));
	printf("\t\t\tIBS OpVal: %d\n",!!(msrs[0]&1ULL<<18));
	printf("\t\t\tIBS OpEn: %d\n",!!(msrs[0]&1ULL<<17));
	printf("\t\t\tIbsOpMaxCnt: %lld\n",((msrs[0]&0xffff)<<4) |
				(msrs[0]&0x3f00000));

	printf("\t\tMSR IBS_OP_LOGICAL_ADDRESS %llx\n",msrs[1]);

	printf("\t\tMSR IBS_OP_DATA %llx\n",msrs[2]);
	printf("\t\t\tRIP Invalid: %d\n",!!(msrs[2]&1ULL<<38));
	printf("\t\t\tBranch Retired: %d\n",!!(msrs[2]&1ULL<<37));
	printf("\t\t\tBranch Mispredicted: %d\n",!!(msrs[2]&1ULL<<36));
	printf("\t\t\tBranch Taken: %d\n",!!(msrs[2]&1ULL<<35));
	printf("\t\t\tReturn uop: %d\n",!!(msrs[2]&1ULL<<34));
	printf("\t\t\tMispredicted Return uop: %d\n",!!(msrs[2]&1ULL<<33));
	printf("\t\t\tTag to Retire Cycles: %lld\n",(msrs[2]>>16)&0xffff);
	printf("\t\t\tCompletion to Retire Cycles: %lld\n",msrs[2]&0xffff);

	printf("\t\tMSR IBS_OP_DATA2 (Northbridge) %llx\n",msrs[3]);
	printf("\t\t\tCache Hit State: %c\n",(msrs[3]&1ULL<<5)?'O':'M');
	printf("\t\t\tRequest destination node: %s\n",
		(msrs[3]&1ULL<<4)?"Same":"Different");
	printf("\t\t\tNorthbridge data source: ");
	switch(msrs[3]&0x7) {
		case 0:	printf("No valid status\n"); break;
		case 1: printf("L3\n"); break;
		case 2: printf("Cache from another compute unit\n"); break;
		case 3: printf("DRAM\n"); break;
		case 4: printf("Reserved remote cache\n"); break;
		case 5: printf("Reserved\n"); break;
		case 6: printf("Reserved\n"); break;
		case 7: printf("Other: MMIO/config/PCI/APIC\n"); break;
	}

	printf("\t\tMSR IBS_OP_DATA3 (cache) %llx\n",msrs[4]);
	printf("\t\t\tData Cache Miss Latency: %lld\n",
		(msrs[4]>>32)&0xffff);
	printf("\t\t\tL2TLB data hit in 1GB page: %d\n",
		!!(msrs[4]&1ULL<<19));
	printf("\t\t\tData cache physical addr valid: %d\n",
		!!(msrs[4]&1ULL<<18));
	printf("\t\t\tData cache linear addr valid: %d\n",
		!!(msrs[4]&1ULL<<17));
	printf("\t\t\tMAB hit: %d\n",
		!!(msrs[4]&1ULL<<16));
	printf("\t\t\tData cache locked operation: %d\n",
		!!(msrs[4]&1ULL<<15));
	printf("\t\t\tUncachable memory operation: %d\n",
		!!(msrs[4]&1ULL<<14));
	printf("\t\t\tWrite-combining memory operation: %d\n",
		!!(msrs[4]&1ULL<<13));
	printf("\t\t\tData forwarding store to load canceled: %d\n",
		!!(msrs[4]&1ULL<<12));
	printf("\t\t\tData forwarding store to load operation: %d\n",
		!!(msrs[4]&1ULL<<11));
	printf("\t\t\tBank conflict on load operation: %d\n",
		!!(msrs[4]&1ULL<<9));
	printf("\t\t\tMisaligned access: %d\n",
		!!(msrs[4]&1ULL<<8));
	printf("\t\t\tData cache miss: %d\n",
		!!(msrs[4]&1ULL<<7));
	printf("\t\t\tData cache L2TLB hit in 2M: %d\n",
		!!(msrs[4]&1ULL<<6));
	printf("\t\t\tData cache L2TLB hit in 1G: %d\n",
		!!(msrs[4]&1ULL<<5));
	printf("\t\t\tData cache L1TLB hit in 2M: %d\n",
		!!(msrs[4]&1ULL<<4));
	printf("\t\t\tData cache L2TLB miss: %d\n",
		!!(msrs[4]&1ULL<<3));
	printf("\t\t\tData cache L1TLB miss: %d\n",
		!!(msrs[4]&1ULL<<2));
	printf("\t\t\tOperation is a store: %d\n",
		!!(msrs[4]&1ULL<<1));
	printf("\t\t\tOperation is a load: %d\n",
		!!(msrs[4]&1ULL<<0));

	if (msrs[4]&1ULL<<17) {
		printf("\t\tMSR IBS_DC_LINEAR_ADDRESS %llx\n",msrs[5]);
	}
	if (msrs[4]&1ULL<<18) {
		printf("\t\tMSR IBS_DC_PHYSICAL_ADDRESS %llx\n",msrs[6]);
	}

	if (size>64) {
		printf("\t\tMSR IBS_OP_DATA4 %llx\n",msrs[7]);
	}
	return 0;
}

static int debug=1;

long long perf_mmap_read( void *our_mmap, int mmap_size,
			long long prev_head,
			int sample_type, int read_format,
			long long reg_mask,
			int quiet, int *events_read,
			int raw_type ) {

	struct perf_event_mmap_page *control_page = our_mmap;
	long long head,offset;
	int i,size;
	long long bytesize,prev_head_wrap;

	unsigned char *data;

	void *data_mmap=our_mmap+getpagesize();

	if (mmap_size==0) return 0;

	if (control_page==NULL) {
		fprintf(stderr,"ERROR mmap page NULL\n");
		return -1;
	}

	head=control_page->data_head;
	rmb(); /* Must always follow read of data_head */

	size=head-prev_head;

	if (debug) {
		printf("Head: %lld Prev_head=%lld\n",head,prev_head);
		printf("%d new bytes\n",size);
	}

	bytesize=mmap_size*getpagesize();

	if (size>bytesize) {
		printf("error!  we overflowed the mmap buffer %d>%lld bytes\n",
			size,bytesize);
	}

	data=malloc(bytesize);
	if (data==NULL) {
		return -1;
	}

	if (debug) {
		printf("Allocated %lld bytes at %p\n",bytesize,data);
	}

	prev_head_wrap=prev_head%bytesize;

	if (debug) {
		printf("Copying %lld bytes from (%p)+%lld to (%p)+%d\n",
			  bytesize-prev_head_wrap,data_mmap,prev_head_wrap,data,0);
	}

	memcpy(data,(unsigned char*)data_mmap + prev_head_wrap,
		bytesize-prev_head_wrap);

	if (debug) {
		printf("Copying %lld bytes from %d to %lld\n",
			prev_head_wrap,0,bytesize-prev_head_wrap);
	}

	memcpy(data+(bytesize-prev_head_wrap),(unsigned char *)data_mmap,
		prev_head_wrap);

	struct perf_event_header *event;


	offset=0;
	if (events_read) *events_read=0;

	while(offset<size) {

		if (debug) printf("Offset %lld Size %d\n",offset,size);
		event = ( struct perf_event_header * ) & data[offset];

		/********************/
		/* Print event Type */
		/********************/

		if (!quiet) {
			switch(event->type) {
				case PERF_RECORD_MMAP:
					printf("PERF_RECORD_MMAP");
					break;
				case PERF_RECORD_LOST:
					printf("PERF_RECORD_LOST");
					break;
				case PERF_RECORD_COMM:
					printf("PERF_RECORD_COMM");
					break;
				case PERF_RECORD_EXIT:
					printf("PERF_RECORD_EXIT");
					break;
				case PERF_RECORD_THROTTLE:
					printf("PERF_RECORD_THROTTLE");
					break;
				case PERF_RECORD_UNTHROTTLE:
					printf("PERF_RECORD_UNTHROTTLE");
					break;
				case PERF_RECORD_FORK:
					printf("PERF_RECORD_FORK");
					break;
				case PERF_RECORD_READ:
					printf("PERF_RECORD_READ");
					break;
				case PERF_RECORD_SAMPLE:
					printf("PERF_RECORD_SAMPLE [%x]",sample_type);
					break;
				case PERF_RECORD_MMAP2:
					printf("PERF_RECORD_MMAP2");
					break;
				case PERF_RECORD_AUX:
					printf("PERF_RECORD_AUX");
					break;
				case PERF_RECORD_ITRACE_START:
					printf("PERF_RECORD_ITRACE_START");
					break;
				case PERF_RECORD_LOST_SAMPLES:
					printf("PERF_RECORD_LOST_SAMPLES");
					break;
				case PERF_RECORD_SWITCH:
					printf("PERF_RECORD_SWITCH");
					break;
				case PERF_RECORD_SWITCH_CPU_WIDE:
					printf("PERF_RECORD_SWITCH_CPU_WIDE");
					break;
				default: printf("UNKNOWN %d",event->type);
					break;
			}

			printf(", MISC=%d (",event->misc);
			switch(event->misc & PERF_RECORD_MISC_CPUMODE_MASK) {
				case PERF_RECORD_MISC_CPUMODE_UNKNOWN:
					printf("PERF_RECORD_MISC_CPUMODE_UNKNOWN"); break; 
				case PERF_RECORD_MISC_KERNEL:
					printf("PERF_RECORD_MISC_KERNEL"); break;
				case PERF_RECORD_MISC_USER:
					printf("PERF_RECORD_MISC_USER"); break;
				case PERF_RECORD_MISC_HYPERVISOR:
					printf("PERF_RECORD_MISC_HYPERVISOR"); break;
				case PERF_RECORD_MISC_GUEST_KERNEL:
					printf("PERF_RECORD_MISC_GUEST_KERNEL"); break;
				case PERF_RECORD_MISC_GUEST_USER:
					printf("PERF_RECORD_MISC_GUEST_USER"); break;
				default:
					printf("Unknown %d!\n",event->misc); break;
			}

			/* All three have the same value */
			if (event->misc & PERF_RECORD_MISC_MMAP_DATA) {
				if (event->type==PERF_RECORD_MMAP) {
					printf(",PERF_RECORD_MISC_MMAP_DATA ");
				}
				else if (event->type==PERF_RECORD_COMM) {
					printf(",PERF_RECORD_MISC_COMM_EXEC ");
				}
				else if ((event->type==PERF_RECORD_SWITCH) ||
					(event->type==PERF_RECORD_SWITCH_CPU_WIDE)) {
					printf(",PERF_RECORD_MISC_SWITCH_OUT ");
				}
				else {
					printf("UNKNOWN ALIAS!!! ");
				}
			}


			if (event->misc & PERF_RECORD_MISC_EXACT_IP) {
				printf(",PERF_RECORD_MISC_EXACT_IP ");
			}

			if (event->misc & PERF_RECORD_MISC_EXT_RESERVED) {
				printf(",PERF_RECORD_MISC_EXT_RESERVED ");
			}

			printf("), Size=%d\n",event->size);
		}

		offset+=8; /* skip header */

		/***********************/
		/* Print event Details */
		/***********************/

		switch(event->type) {

		/* Lost */
		case PERF_RECORD_LOST: {
			long long id,lost;
			memcpy(&id,&data[offset],sizeof(long long));
			if (!quiet) printf("\tID: %lld\n",id);
			offset+=8;
			memcpy(&lost,&data[offset],sizeof(long long));
			if (!quiet) printf("\tLOST: %lld\n",lost);
			offset+=8;
			}
			break;

		/* COMM */
		case PERF_RECORD_COMM: {
			int pid,tid,string_size;
			char *string;

			memcpy(&pid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPID: %d\n",pid);
			offset+=4;
			memcpy(&tid,&data[offset],sizeof(int));
			if (!quiet) printf("\tTID: %d\n",tid);
			offset+=4;

			/* FIXME: sample_id handling? */

			/* two ints plus the 64-bit header */
			string_size=event->size-16;
			string=calloc(string_size,sizeof(char));
			memcpy(string,&data[offset],string_size);
			if (!quiet) printf("\tcomm: %s\n",string);
			offset+=string_size;
			if (string) free(string);
			}
			break;

		/* Fork */
		case PERF_RECORD_FORK: {
			int pid,ppid,tid,ptid;
			long long fork_time;

			memcpy(&pid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPID: %d\n",pid);
			offset+=4;
			memcpy(&ppid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPPID: %d\n",ppid);
			offset+=4;
			memcpy(&tid,&data[offset],sizeof(int));
			if (!quiet) printf("\tTID: %d\n",tid);
			offset+=4;
			memcpy(&ptid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPTID: %d\n",ptid);
			offset+=4;
			memcpy(&fork_time,&data[offset],sizeof(long long));
			if (!quiet) printf("\tTime: %lld\n",fork_time);
			offset+=8;
			}
			break;

		/* mmap */
		case PERF_RECORD_MMAP: {
			int pid,tid,string_size;
			long long address,len,pgoff;
			char *filename;

			memcpy(&pid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPID: %d\n",pid);
			offset+=4;
			memcpy(&tid,&data[offset],sizeof(int));
			if (!quiet) printf("\tTID: %d\n",tid);
			offset+=4;
			memcpy(&address,&data[offset],sizeof(long long));
			if (!quiet) printf("\tAddress: %llx\n",address);
			offset+=8;
			memcpy(&len,&data[offset],sizeof(long long));
			if (!quiet) printf("\tLength: %llx\n",len);
			offset+=8;
			memcpy(&pgoff,&data[offset],sizeof(long long));
			if (!quiet) printf("\tPage Offset: %llx\n",pgoff);
			offset+=8;

			string_size=event->size-40;
			filename=calloc(string_size,sizeof(char));
			memcpy(filename,&data[offset],string_size);
			if (!quiet) printf("\tFilename: %s\n",filename);
			offset+=string_size;
			if (filename) free(filename);

			}
			break;

		/* mmap2 */
		case PERF_RECORD_MMAP2: {
			int pid,tid,string_size;
			long long address,len,pgoff;
			int major,minor;
			long long ino,ino_generation;
			int prot,flags;
			char *filename;

			memcpy(&pid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPID: %d\n",pid);
			offset+=4;
			memcpy(&tid,&data[offset],sizeof(int));
			if (!quiet) printf("\tTID: %d\n",tid);
			offset+=4;
			memcpy(&address,&data[offset],sizeof(long long));
			if (!quiet) printf("\tAddress: %llx\n",address);
			offset+=8;
			memcpy(&len,&data[offset],sizeof(long long));
			if (!quiet) printf("\tLength: %llx\n",len);
			offset+=8;
			memcpy(&pgoff,&data[offset],sizeof(long long));
			if (!quiet) printf("\tPage Offset: %llx\n",pgoff);
			offset+=8;
			memcpy(&major,&data[offset],sizeof(int));
			if (!quiet) printf("\tMajor: %d\n",major);
			offset+=4;
			memcpy(&minor,&data[offset],sizeof(int));
			if (!quiet) printf("\tMinor: %d\n",minor);
			offset+=4;
			memcpy(&ino,&data[offset],sizeof(long long));
			if (!quiet) printf("\tIno: %llx\n",ino);
			offset+=8;
			memcpy(&ino_generation,&data[offset],sizeof(long long));
			if (!quiet) printf("\tIno generation: %llx\n",ino_generation);
			offset+=8;
			memcpy(&prot,&data[offset],sizeof(int));
			if (!quiet) printf("\tProt: %d\n",prot);
			offset+=4;
			memcpy(&flags,&data[offset],sizeof(int));
			if (!quiet) printf("\tFlags: %d\n",flags);
			offset+=4;

			string_size=event->size-72;
			filename=calloc(string_size,sizeof(char));
			memcpy(filename,&data[offset],string_size);
			if (!quiet) printf("\tFilename: %s\n",filename);
			offset+=string_size;
			if (filename) free(filename);

			}
			break;

		/* Exit */
		case PERF_RECORD_EXIT: {
			int pid,ppid,tid,ptid;
			long long fork_time;

			memcpy(&pid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPID: %d\n",pid);
			offset+=4;
			memcpy(&ppid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPPID: %d\n",ppid);
			offset+=4;
			memcpy(&tid,&data[offset],sizeof(int));
			if (!quiet) printf("\tTID: %d\n",tid);
			offset+=4;
			memcpy(&ptid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPTID: %d\n",ptid);
			offset+=4;
			memcpy(&fork_time,&data[offset],sizeof(long long));
			if (!quiet) printf("\tTime: %lld\n",fork_time);
			offset+=8;
			}
			break;

		/* Throttle/Unthrottle */
		case PERF_RECORD_THROTTLE:
		case PERF_RECORD_UNTHROTTLE: {
			long long throttle_time,id,stream_id;

			memcpy(&throttle_time,&data[offset],sizeof(long long));
			if (!quiet) printf("\tTime: %lld\n",throttle_time);
			offset+=8;
			memcpy(&id,&data[offset],sizeof(long long));
			if (!quiet) printf("\tID: %lld\n",id);
			offset+=8;
			memcpy(&stream_id,&data[offset],sizeof(long long));
			if (!quiet) printf("\tStream ID: %lld\n",stream_id);
			offset+=8;

			}
			break;

		/* Sample */
		case PERF_RECORD_SAMPLE:
			if (sample_type & PERF_SAMPLE_IP) {
				long long ip;
				memcpy(&ip,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_IP, IP: %llx\n",ip);
				offset+=8;
			}
			if (sample_type & PERF_SAMPLE_TID) {
				int pid, tid;
				memcpy(&pid,&data[offset],sizeof(int));
				memcpy(&tid,&data[offset+4],sizeof(int));

				if (!quiet) {
					printf("\tPERF_SAMPLE_TID, pid: %d  tid %d\n",pid,tid);
				}
				offset+=8;
			}

			if (sample_type & PERF_SAMPLE_TIME) {
				long long time;
				memcpy(&time,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_TIME, time: %lld\n",time);
				offset+=8;
			}
			if (sample_type & PERF_SAMPLE_ADDR) {
				long long addr;
				memcpy(&addr,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_ADDR, addr: %llx\n",addr);
				offset+=8;
			}
			if (sample_type & PERF_SAMPLE_ID) {
				long long sample_id;
				memcpy(&sample_id,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_ID, sample_id: %lld\n",sample_id);
				offset+=8;
			}
			if (sample_type & PERF_SAMPLE_STREAM_ID) {
				long long sample_stream_id;
				memcpy(&sample_stream_id,&data[offset],sizeof(long long));
				if (!quiet) {
					printf("\tPERF_SAMPLE_STREAM_ID, sample_stream_id: %lld\n",sample_stream_id);
				}
				offset+=8;
			}
			if (sample_type & PERF_SAMPLE_CPU) {
				int cpu, res;
				memcpy(&cpu,&data[offset],sizeof(int));
				memcpy(&res,&data[offset+4],sizeof(int));
				if (!quiet) printf("\tPERF_SAMPLE_CPU, cpu: %d  res %d\n",cpu,res);
				offset+=8;
			}
			if (sample_type & PERF_SAMPLE_PERIOD) {
				long long period;
				memcpy(&period,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_PERIOD, period: %lld\n",period);
				offset+=8;
			}
			if (sample_type & PERF_SAMPLE_READ) {
				int length;

				if (!quiet) printf("\tPERF_SAMPLE_READ, read_format\n");
				length=handle_struct_read_format(&data[offset],
						read_format,
						quiet);
				if (length>=0) offset+=length;
			}
			if (sample_type & PERF_SAMPLE_CALLCHAIN) {
				long long nr,ip;
				memcpy(&nr,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_CALLCHAIN, callchain length: %lld\n",nr);
				offset+=8;

				for(i=0;i<nr;i++) {
					memcpy(&ip,&data[offset],sizeof(long long));
					if (!quiet) printf("\t\t ip[%d]: %llx\n",i,ip);
					offset+=8;
				}
			}
			if (sample_type & PERF_SAMPLE_RAW) {
				int size;

				memcpy(&size,&data[offset],sizeof(int));
				if (!quiet) printf("\tPERF_SAMPLE_RAW, Raw length: %d\n",size);
				offset+=4;

				if (!quiet) {
					if (raw_type==RAW_IBS_FETCH) {
						dump_raw_ibs_fetch(&data[offset],size);
					}
					else if (raw_type==RAW_IBS_OP) {
						dump_raw_ibs_op(&data[offset],size);
					}
					else {
						printf("\t\t");
						for(i=0;i<size;i++) {
							printf("%d ",data[offset+i]);
						}
						printf("\n");
					}
				}
				offset+=size;
			}

			if (sample_type & PERF_SAMPLE_BRANCH_STACK) {
				long long bnr;
				memcpy(&bnr,&data[offset],sizeof(long long));
				if (!quiet) {
					printf("\tPERF_SAMPLE_BRANCH_STACK, branch_stack entries: %lld\n",bnr);
				}
				offset+=8;

				for(i=0;i<bnr;i++) {
					long long from,to,flags;

					/* From value */
					memcpy(&from,&data[offset],sizeof(long long));
					offset+=8;


					/* To Value */
					memcpy(&to,&data[offset],sizeof(long long));
					offset+=8;
					if (!quiet) {
						printf("\t\t lbr[%d]: %llx %llx ",
							i,from,to);
		 			}

					/* Flags */
					memcpy(&flags,&data[offset],sizeof(long long));
					offset+=8;

					if (!quiet) {
						if (flags==0) printf("0");

						if (flags&1) {
							printf("MISPREDICTED ");
							flags&=~2;
						}

						if (flags&2) {
							printf("PREDICTED ");
							flags&=~2;
						}

						if (flags&4) {
							printf("IN_TRANSACTION ");
							flags&=~4;
						}

						if (flags&8) {
							printf("TRANSACTION_ABORT ");
							flags&=~8;
						}
						printf("\n");
					}
	      			}
	   		}

			if (sample_type & PERF_SAMPLE_REGS_USER) {
				long long abi;

				memcpy(&abi,&data[offset],sizeof(long long));
				if (!quiet) {
					printf("\tPERF_SAMPLE_REGS_USER, ABI: ");
					if (abi==PERF_SAMPLE_REGS_ABI_NONE) printf ("PERF_SAMPLE_REGS_ABI_NONE");
					if (abi==PERF_SAMPLE_REGS_ABI_32) printf("PERF_SAMPLE_REGS_ABI_32");
					if (abi==PERF_SAMPLE_REGS_ABI_64) printf("PERF_SAMPLE_REGS_ABI_64");
					printf("\n");
				}
				offset+=8;

				offset+=print_regs(quiet,abi,reg_mask,
						&data[offset]);

				if (!quiet) printf("\n");
			}

			if (sample_type & PERF_SAMPLE_REGS_INTR) {
				long long abi;

				memcpy(&abi,&data[offset],sizeof(long long));
				if (!quiet) {
					printf("\tPERF_SAMPLE_REGS_INTR, ABI: ");
					if (abi==PERF_SAMPLE_REGS_ABI_NONE) printf ("PERF_SAMPLE_REGS_ABI_NONE");
					if (abi==PERF_SAMPLE_REGS_ABI_32) printf("PERF_SAMPLE_REGS_ABI_32");
					if (abi==PERF_SAMPLE_REGS_ABI_64) printf("PERF_SAMPLE_REGS_ABI_64");
					printf("\n");
				}
				offset+=8;

				offset+=print_regs(quiet,abi,reg_mask,
						&data[offset]);

				if (!quiet) printf("\n");
			}

			if (sample_type & PERF_SAMPLE_STACK_USER) {
				long long size,dyn_size;
				int *stack_data;
				int k;

				memcpy(&size,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_STACK_USER, Requested size: %lld\n",size);
				offset+=8;

				stack_data=malloc(size);
				memcpy(stack_data,&data[offset],size);
				offset+=size;

				memcpy(&dyn_size,&data[offset],sizeof(long long));
				if (!quiet) printf("\t\tDynamic (used) size: %lld\n",dyn_size);
				offset+=8;

				if (!quiet) printf("\t\t");
				for(k=0;k<dyn_size;k+=4) {
					if (!quiet) printf("0x%x ",stack_data[k]);
				}

				free(stack_data);

				if (!quiet) printf("\n");
			}

			if (sample_type & PERF_SAMPLE_WEIGHT) {
				long long weight;

				memcpy(&weight,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_WEIGHT, Weight: %lld ",weight);
				offset+=8;

				if (!quiet) printf("\n");
			}

			if (sample_type & PERF_SAMPLE_DATA_SRC) {
				long long src;

				memcpy(&src,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_DATA_SRC, Raw: %llx\n",src);
				offset+=8;

				if (!quiet) {
					if (src!=0) printf("\t\t");
					if (src & (PERF_MEM_OP_NA<<PERF_MEM_OP_SHIFT))
						printf("Op Not available ");
					if (src & (PERF_MEM_OP_LOAD<<PERF_MEM_OP_SHIFT))
						printf("Load ");
					if (src & (PERF_MEM_OP_STORE<<PERF_MEM_OP_SHIFT))
						printf("Store ");
					if (src & (PERF_MEM_OP_PFETCH<<PERF_MEM_OP_SHIFT))
						printf("Prefetch ");
					if (src & (PERF_MEM_OP_EXEC<<PERF_MEM_OP_SHIFT))
						printf("Executable code ");

					if (src & (PERF_MEM_LVL_NA<<PERF_MEM_LVL_SHIFT))
						printf("Level Not available ");
					if (src & (PERF_MEM_LVL_HIT<<PERF_MEM_LVL_SHIFT))
						printf("Hit ");
					if (src & (PERF_MEM_LVL_MISS<<PERF_MEM_LVL_SHIFT))
						printf("Miss ");
					if (src & (PERF_MEM_LVL_L1<<PERF_MEM_LVL_SHIFT))
						printf("L1 cache ");
					if (src & (PERF_MEM_LVL_LFB<<PERF_MEM_LVL_SHIFT))
						printf("Line fill buffer ");
					if (src & (PERF_MEM_LVL_L2<<PERF_MEM_LVL_SHIFT))
						printf("L2 cache ");
					if (src & (PERF_MEM_LVL_L3<<PERF_MEM_LVL_SHIFT))
						printf("L3 cache ");
					if (src & (PERF_MEM_LVL_LOC_RAM<<PERF_MEM_LVL_SHIFT))
						printf("Local DRAM ");
					if (src & (PERF_MEM_LVL_REM_RAM1<<PERF_MEM_LVL_SHIFT))
						printf("Remote DRAM 1 hop ");
					if (src & (PERF_MEM_LVL_REM_RAM2<<PERF_MEM_LVL_SHIFT))
						printf("Remote DRAM 2 hops ");
					if (src & (PERF_MEM_LVL_REM_CCE1<<PERF_MEM_LVL_SHIFT))
						printf("Remote cache 1 hop ");
					if (src & (PERF_MEM_LVL_REM_CCE2<<PERF_MEM_LVL_SHIFT))
						printf("Remote cache 2 hops ");
					if (src & (PERF_MEM_LVL_IO<<PERF_MEM_LVL_SHIFT))
						printf("I/O memory ");
					if (src & (PERF_MEM_LVL_UNC<<PERF_MEM_LVL_SHIFT))
						printf("Uncached memory ");

					if (src & (PERF_MEM_SNOOP_NA<<PERF_MEM_SNOOP_SHIFT))
						printf("Not available ");
					if (src & (PERF_MEM_SNOOP_NONE<<PERF_MEM_SNOOP_SHIFT))
						printf("No snoop ");
					if (src & (PERF_MEM_SNOOP_HIT<<PERF_MEM_SNOOP_SHIFT))
						printf("Snoop hit ");
					if (src & (PERF_MEM_SNOOP_MISS<<PERF_MEM_SNOOP_SHIFT))
						printf("Snoop miss ");
					if (src & (PERF_MEM_SNOOP_HITM<<PERF_MEM_SNOOP_SHIFT))
						printf("Snoop hit modified ");

					if (src & (PERF_MEM_LOCK_NA<<PERF_MEM_LOCK_SHIFT))
						printf("Not available ");
					if (src & (PERF_MEM_LOCK_LOCKED<<PERF_MEM_LOCK_SHIFT))
						printf("Locked transaction ");

					if (src & (PERF_MEM_TLB_NA<<PERF_MEM_TLB_SHIFT))
						printf("Not available ");
					if (src & (PERF_MEM_TLB_HIT<<PERF_MEM_TLB_SHIFT))
						printf("Hit ");
					if (src & (PERF_MEM_TLB_MISS<<PERF_MEM_TLB_SHIFT))
						printf("Miss ");
					if (src & (PERF_MEM_TLB_L1<<PERF_MEM_TLB_SHIFT))
						printf("Level 1 TLB ");
					if (src & (PERF_MEM_TLB_L2<<PERF_MEM_TLB_SHIFT))
						printf("Level 2 TLB ");
					if (src & (PERF_MEM_TLB_WK<<PERF_MEM_TLB_SHIFT))
						printf("Hardware walker ");
					if (src & ((long long)PERF_MEM_TLB_OS<<PERF_MEM_TLB_SHIFT))
						printf("OS fault handler ");
				}

				if (!quiet) printf("\n");
			}

			if (sample_type & PERF_SAMPLE_IDENTIFIER) {
				long long abi;

				memcpy(&abi,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_IDENTIFIER, Raw length: %lld\n",abi);
				offset+=8;

				if (!quiet) printf("\n");
			}

			if (sample_type & PERF_SAMPLE_TRANSACTION) {
				long long abi;

				memcpy(&abi,&data[offset],sizeof(long long));
				if (!quiet) printf("\tPERF_SAMPLE_TRANSACTION, Raw length: %lld\n",abi);
				offset+=8;

				if (!quiet) printf("\n");
			}
			break;

		/* AUX */
		case PERF_RECORD_AUX: {
			long long aux_offset,aux_size,flags;
			long long sample_id;

			memcpy(&aux_offset,&data[offset],sizeof(long long));
			if (!quiet) printf("\tAUX_OFFSET: %lld\n",aux_offset);
			offset+=8;

			memcpy(&aux_size,&data[offset],sizeof(long long));
			if (!quiet) printf("\tAUX_SIZE: %lld\n",aux_size);
			offset+=8;

			memcpy(&flags,&data[offset],sizeof(long long));
			if (!quiet) {
				printf("\tFLAGS: %llx ",flags);
				if (flags & PERF_AUX_FLAG_TRUNCATED) {
					printf("FLAG_TRUNCATED ");
				}
				if (flags & PERF_AUX_FLAG_OVERWRITE) {
					printf("FLAG_OVERWRITE ");
				}
				printf("\n");
			}
			offset+=8;

			memcpy(&sample_id,&data[offset],sizeof(long long));
			if (!quiet) printf("\tSAMPLE_ID: %lld\n",sample_id);
			offset+=8;

			}
			break;

		/* itrace start */
		case PERF_RECORD_ITRACE_START: {
			int pid,tid;

			memcpy(&pid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPID: %d\n",pid);
			offset+=4;

			memcpy(&tid,&data[offset],sizeof(int));
			if (!quiet) printf("\tTID: %d\n",tid);
			offset+=4;
			}
			break;

		/* lost samples PEBS */
		case PERF_RECORD_LOST_SAMPLES: {
			long long lost,sample_id;

			memcpy(&lost,&data[offset],sizeof(long long));
			if (!quiet) printf("\tLOST: %lld\n",lost);
			offset+=8;

			memcpy(&sample_id,&data[offset],sizeof(long long));
			if (!quiet) printf("\tSAMPLE_ID: %lld\n",sample_id);
			offset+=8;
			}
			break;

		/* context switch */
		case PERF_RECORD_SWITCH: {
			long long sample_id;

			memcpy(&sample_id,&data[offset],sizeof(long long));
			if (!quiet) printf("\tSAMPLE_ID: %lld\n",sample_id);
			offset+=8;
			}
			break;

		/* context switch cpu-wide*/
		case PERF_RECORD_SWITCH_CPU_WIDE: {
			int prev_pid,prev_tid;
			long long sample_id;

			memcpy(&prev_pid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPREV_PID: %d\n",prev_pid);
			offset+=4;

			memcpy(&prev_tid,&data[offset],sizeof(int));
			if (!quiet) printf("\tPREV_TID: %d\n",prev_tid);
			offset+=4;

			memcpy(&sample_id,&data[offset],sizeof(long long));
			if (!quiet) printf("\tSAMPLE_ID: %lld\n",sample_id);
			offset+=8;
			}
			break;


		default:
			if (!quiet) printf("\tUnknown type %d\n",event->type);
			/* Probably best to just skip it all */
			offset=size;

		}
		if (events_read) (*events_read)++;
	}

//	mb();
	control_page->data_tail=head;

	free(data);

	return head;

}


void
handler( int EventSet, void *address, long long overflow_vector, void *context )
{
	( void ) context;

	prev_head=perf_mmap_read(address,
			MMAP_DATA_SIZE,
			prev_head,
			sample_type,0,
			global_sample_regs_user,
			quiet,NULL,RAW_NONE);

	if ( !TESTS_QUIET ) {
		fprintf( stderr, "handler(%d ) Overflow at %p! bit=%#llx \n",
			EventSet, address, overflow_vector );
	}
	total++;
}

#define NUM_EVENTS	1

int main(int argc, char **argv ) {

	int EventSet = PAPI_NULL;
	long long values_original[NUM_EVENTS];
	long long values_sampled[NUM_EVENTS];
	//long long min, max;
	int num_flops = NUM_FLOPS, retval;
	int PAPI_event, mythreshold = THRESHOLD;
	char event_name1[PAPI_MAX_STR_LEN];
	const PAPI_hw_info_t *hw_info = NULL;
	int result;

	/* Set TESTS_QUIET variable */
	quiet = tests_quiet( argc, argv );

	/* Init PAPI */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if ( retval != PAPI_VER_CURRENT ) {
		test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
	}

	/* Get hardware info */
	hw_info = PAPI_get_hardware_info(  );
	if ( hw_info == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_get_hardware_info", 2 );
	}

	/* Set up eventset */
	result=PAPI_create_eventset( &EventSet );
	if ( hw_info == NULL ) {
		test_fail( __FILE__, __LINE__, "PAPI_create_Eventset", 2 );
	}

	/* add PAPI_TOT_CYC */
	result = PAPI_add_named_event(EventSet, "PAPI_TOT_CYC");
	if (result!=PAPI_OK) {
		if (!quiet) {
			printf("Trouble adding event! %s\n",
					PAPI_strerror(result));
		}
		test_skip(__FILE__,__LINE__,"Event add",1);
	}

	if (!quiet) {
		printf("Using PAPI_TOT_CYC for the overflow event\n");
	}

	result=PAPI_event_name_to_code("PAPI_TOT_CYC",&PAPI_event);
	if (result!=PAPI_OK) {
		if (!quiet) printf("Trouble converting to code\n");
		test_skip(__FILE__,__LINE__,"Event name to cod",1);
	}

	mythreshold=100000;

	/* Start the run calibration run */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_start", retval );

	do_flops( NUM_FLOPS );

	/* stop the calibration run */
	retval = PAPI_stop( EventSet, values_original );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );


	/* set up sample handler */
#if 0
	retval = PAPI_overflow( EventSet,
				PAPI_event,
				mythreshold, 0, handler );
#endif


	sample_type = 		PERF_SAMPLE_IP |
				PERF_SAMPLE_WEIGHT |
				PERF_SAMPLE_DATA_SRC;

	retval = PAPI_sample_init( EventSet,
				PAPI_event,
				mythreshold,
				sample_type,
				handler );

	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );
	}

	/* Start overflow run */
	retval = PAPI_start( EventSet );
	if ( retval != PAPI_OK ) {
	   test_fail( __FILE__, __LINE__, "PAPI_start", retval );
	}

	do_flops( num_flops );

	/* stop overflow run */
	retval = PAPI_stop( EventSet, values_sampled );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_stop", retval );
	retval = PAPI_overflow( EventSet, PAPI_event, 0, 0, handler );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_overflow", retval );

	if ( !TESTS_QUIET ) {
		retval = PAPI_event_code_to_name( PAPI_event, event_name1 );
		if (retval != PAPI_OK ) {
			test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", retval );
		}

		printf( "Test case: Overflow dispatch of 2nd event in set with 2 events.\n" );
		printf( "---------------------------------------------------------------\n" );
		printf( "Threshold for overflow is: %d\n", mythreshold );
		printf( "Using %d iterations of c += a*b\n", num_flops );
		printf( "-----------------------------------------------\n" );

		printf( "Test type    : %16s%16s\n", "original", "sampling" );
		printf( "%-12s : %16lld%16lld\n", "PAPI_TOT_CYC",
			values_original[0],values_sampled[0]);
		printf( "Overflows    : %16s%16d\n", "", total );
		printf( "-----------------------------------------------\n" );
	}

	retval = PAPI_cleanup_eventset( EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_cleanup_eventset", retval );

	retval = PAPI_destroy_eventset( &EventSet );
	if ( retval != PAPI_OK )
		test_fail( __FILE__, __LINE__, "PAPI_destroy_eventset", retval );

	if ( !TESTS_QUIET ) {
		printf( "Verification:\n" );
#if 0

		if ( PAPI_event == PAPI_FP_INS || PAPI_event == PAPI_FP_OPS ) {
			printf( "Row 1 approximately equals %d %d\n", num_flops,
					num_flops );
		}
		printf( "Column 1 approximately equals column 2\n" );
		printf( "Row 3 approximately equals %u +- %u %%\n",
				( unsigned ) ( ( values[0] )[1] / ( long long ) mythreshold ),
				( unsigned ) ( OVR_TOLERANCE * 100.0 ) );
#endif
	}
/*
  min = (long long)((values[0])[1]*(1.0-TOLERANCE));
  max = (long long)((values[0])[1]*(1.0+TOLERANCE));
  if ( (values[0])[1] > max || (values[0])[1] < min )
  	test_fail(__FILE__, __LINE__, event_name, 1);
*/

#if 0
	min =
		( long long ) ( ( ( double ) values[0][1] * ( 1.0 - OVR_TOLERANCE ) ) /
						( double ) mythreshold );
	max =
		( long long ) ( ( ( double ) values[0][1] * ( 1.0 + OVR_TOLERANCE ) ) /
						( double ) mythreshold );
	if (!quiet) {
		printf( "Overflows: total(%d) > max(%lld) || "
			"total(%d) < min(%lld) \n", total, max, total, min );
	}
	if ( total > max || total < min ) {
		test_fail( __FILE__, __LINE__, "Overflows", 1 );
	}
#endif
	test_pass( __FILE__ );
	return 0;

}

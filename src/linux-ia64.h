#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/times.h>
#include <asm/system.h>
#include "mysiginfo.h"
#include "papi.h"
#include "papi_internal.h"
#include "papiStdEventDefs.h"

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

#include "pfmlib.h"

extern char *basename(char *);
extern caddr_t _init, _fini, _etext, _edata, __bss_start;





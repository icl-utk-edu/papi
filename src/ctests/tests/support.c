/* file: support.c */
/* program to show which standard events are available on the current system*/

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "papi_internal.h"

/* Header files for the substrates */
#if defined(mips) && defined(unix) && defined(sgi)
#include "irix-mips.h"
#elif defined(i386) && defined(unix) && defined(linux)
#include "linux-pentium.h"
#else
#include "any-null.h"
#endif

void main(int argc, char *argv[]) {

typedef struct tag {
	char         name[16];
        unsigned int num;
        } TAG; 

TAG   evt[26] = { "PAPI_L1_DCM"  ,  0x80000000,
		  "PAPI_L1_ICM"  ,  0x80000001,
		  "PAPI_L2_DCM"  ,  0x80000002,
		  "PAPI_L2_ICM"  ,  0x80000003,
	          "PAPI_L3_DCM"  ,  0x80000004,
		  "PAPI_L3_ICM"  ,  0x80000005,
                  "PAPI_CA_SHR"  ,  0x8000000A,
		  "PAPI_CA_CLN"  ,  0x8000000B,
		  "PAPI_CA_INV"  ,  0x8000000C,
		  "PAPI_TLB_DM"  ,  0x80000014,
		  "PAPI_TLB_IM"  ,  0x80000015,
		  "PAPI_TLB_SD"  ,  0x8000001E,
		  "PAPI_BRI_MSP" ,  0x80000028,
		  "PAPI_BRI_TKN" ,  0x80000029,
		  "PAPI_BRI_NTK" ,  0x8000002A,
		  "PAPI_TOT_INS" ,  0x80000032,
		  "PAPI_INT_INS" ,  0x80000033,
		  "PAPI_FP_INS"  ,  0x80000034,
		  "PAPI_LD_INS"  ,  0x80000035,
		  "PAPI_SR_INS"  ,  0x80000036,
		  "PAPI_CND_INS" ,  0x80000037,
		  "PAPI_VEC_INS" ,  0x80000038,
		  "PAPI_FLOPS"   ,  0x80000039,
                  "PAPI_TOT_CYC" ,  0x8000003C,
		  "PAPI_MIPS"    ,  0x8000003D   };

char *str1="echo \"Host substrate:  \" ; uname -sr";
char *str2="echo \"Host architecture:  \"; arch"; 
int i;
int r[26]; /* return codes*/
int EventSet[26]; /* integers to reference event sets*/

for(i=0;i<26;i++) EventSet[i]=PAPI_NULL;

printf("\n begin %s:\n show which hardware events have support on current system\n\n",__FILE__);

printf("\n");
for (i=0; i<25; i++) {

    r[i]=PAPI_add_event(&EventSet[i],evt[i].num);
}

printf("\n");
i=0;
while(i<25) {
    if(r[i]<PAPI_OK)
    printf(" %13s not supported here\n", evt[i].name);
    i++;
}
printf("\n");
i=0;
while(i<25) {
    if(r[i]>=PAPI_OK)             
    printf(" %13s found \n", evt[i].name); 
    i++;
}
printf("\n---------------------------------------");
printf("\nMax number of events in one EventSet: %d",
        _papi_system_info.num_gp_cntrs
      +_papi_system_info.num_sp_cntrs   );
printf("\n---------------------------------------");
system(str2);
system(str1);
printf("\n");

PAPI_shutdown();
printf(" normal termination %s\n",__FILE__);
}/* end main*/    

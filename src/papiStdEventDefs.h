/* file: papiStdEventDefs.h

The following is a list of hardware events deemed relevant and useful in tuning
application performance. These events have identical assignments in the header
files on different platforms however they may differ in their actual semantics. In
addition, all of these events are not guaranteed to be present on all platforms.
Please check your platform's documentation carefully. 

This file should be modified to comply with the documentation on 
your platform.
 
*/

#define PAPI_L1_DCM  0x80000000 /*Level 1 data cache misses*/
#define PAPI_L1_ICM  0x80000001 /*Level 1 instruction cache misses*/ 
#define PAPI_L2_DCM  0x80000002 /*Level 2 data cache misses*/
#define PAPI_L2_ICM  0x80000003 /*Level 2 instruction cache misses*/ 
#define PAPI_L3_DCM  0x80000004 /*Level 3 data cache misses*/
#define PAPI_L3_ICM  0x80000005 /*Level 3 instruction cache misses*/ 

#define PAPI_CA_SHR  0x8000000A /*Request for access to shared cache line (SMP)*/
#define PAPI_CA_CLN  0x8000000B /*Request for access to clean  cache line (SMP)*/
#define PAPI_CA_INV  0x8000000C /*Cache Line Invalidation (SMP)*/

#define PAPI_TLB_DM  0x80000014 /*Data translation lookaside buffer misses*/
#define PAPI_TLB_IM  0x80000015 /*Instruction translation lookaside buffer misses*/
#define PAPI_TLB_SD  0x8000001E /*Translation lookaside buffer shootdowns (SMP)*/

#define PAPI_BRI_MSP 0x80000028 /*Branch instructions mispredicted*/
#define PAPI_BRI_TKN 0x80000029 /*Branch instructions taken*/
#define PAPI_BRI_NTK 0x8000002A /*Branch instructions not taken*/

#define PAPI_TOT_INS 0x80000032 /*Total instructions executed*/
#define PAPI_INT_INS 0x80000033 /*Integer instructions executed*/
#define PAPI_FP_INS  0x80000034 /*Floating point instructions executed*/
#define PAPI_LD_INS  0x80000035 /*Load instructions executed*/
#define PAPI_SR_INS  0x80000036 /*Store instructions executed*/
#define PAPI_CND_INS 0x80000037 /*Branch instructions executed*/
#define PAPI_VEC_INS 0x80000038 /*Vector/SIMD instructions executed*/
#define PAPI_FLOPS   0x80000039 /*Floating Point instructions per second*/ 

#define PAPI_TOT_CYC 0x8000003C /*Total cycles*/
#define PAPI_MIPS    0x8000003D /*Millions of instructions executed per second*/



   char *standardEventDef_STR[24]= {
	"PAPI_L1_DCM",
	"PAPI_L1_ICM",
	"PAPI_L2_DCM",
	"PAPI_L2_ICM",
	"PAPI_L3_DCM",
	"PAPI_CA_SHR",
	"PAPI_CA_CLN",
	"PAPI_CA_INV",
	"PAPI_TLB_DM",
	"PAPI_TLB_IM",
	"PAPI_TLB_SD",
	"PAPI_BRI_MSP",
	"PAPI_BRI_TKN",
	"PAPI_BRI_NTK",
	"PAPI_TOT_INS",
	"PAPI_INT_INS",
	"PAPI_FP_INS",
	"PAPI_LD_INS",
	"PAPI_SR_INS",
	"PAPI_CND_INS",
	"PAPI_VEC_INS",
	"PAPI_FLOPS",
	"PAPI_TOT_CYc",
	"PAPI_MIPS"      };

   long long standardEventDef_NUM[24]= {
	PAPI_L1_DCM,
	PAPI_L1_ICM,
	PAPI_L2_DCM,
	PAPI_L2_ICM,
	PAPI_L3_DCM,
	PAPI_CA_SHR,
	PAPI_CA_CLN,
	PAPI_CA_INV,
	PAPI_TLB_DM,
	PAPI_TLB_IM,
	PAPI_TLB_SD,
	PAPI_BRI_MSP,
	PAPI_BRI_TKN,
	PAPI_BRI_NTK,
	PAPI_TOT_INS,
	PAPI_INT_INS,
	PAPI_FP_INS,
	PAPI_LD_INS,
	PAPI_SR_INS,
	PAPI_CND_INS,
	PAPI_VEC_INS,
	PAPI_FLOPS,
	PAPI_TOT_CYC,
	PAPI_MIPS      };



/* file: papiStdEventDefs.h

The following is a list of hardware events deemed relevant and useful in tuning
application performance. These events have identical assignments in the header
files on different platforms however they may differ in their actual semantics. In
addition, all of these events are not guaranteed to be present on all platforms.
Please check your platform's documentation carefully. 

This file should be modified to comply with the documentation on 
your platform.
 
*/

#define PAPI_L1_DCM  0x80010000 /*Level 1 data cache misses*/
#define PAPI_L1_ICM  0x80010001 /*Level 1 instruction cache misses*/ 
#define PAPI_L2_DCM  0x80010002 /*Level 2 data cache misses*/
#define PAPI_L2_ICM  0x80010003 /*Level 2 instruction cache misses*/ 
#define PAPI_L3_DCM  0x80010004 /*Level 3 data cache misses*/
#define PAPI_L3_ICM  0x80010005 /*Level 3 instruction cache misses*/ 

#define PAPI_CA_SHR  0x80011000 /*Request for access to shared cache line (SMP)*/
#define PAPI_CA_CLN  0x80011001 /*Request for access to clean  cache line (SMP)*/
#define PAPI_CA_INV  0x80011002 /*Cache Line Invalidation (SMP)*/

#define PAPI_TBL_DM  0x80020000 /*Data translation lookaside buffer misses*/
#define PAPI_TBL_IM  0x80020001 /*Instruction translation lookaside buffer misses*/
#define PAPI_TBL_SD  0x80021000 /*Translation lookaside buffer shootdowns (SMP)*/

#define PAPI_BRI_MSP 0x80030000 /*Branch instructions mispredicted*/
#define PAPI_BRI_TKN 0x80030001 /*Branch instructions taken*/
#define PAPI_BRI_NTK 0x80030002 /*Branch instructions not taken*/

#define PAPI_TOT_INS 0x80100000 /*Total instructions executed*/
#define PAPI_INT_INS 0x80100001 /*Integer instructions executed*/
#define PAPI_FP_INS  0x80100002 /*Floating point instructions executed*/
#define PAPI_LD_INS  0x80100003 /*Load instructions executed*/
#define PAPI_SR_INS  0x80100004 /*Store instructions executed*/
#define PAPI_CND_INS 0x80100005 /*Branch instructions executed*/
#define PAPI_VEC_INS 0x80100006 /*Vector/SIMD instructions executed*/
#define PAPI_FLOPS   0x80100007 /*Floating Point instructions per second*/ 

#define PAPI_TOT_CYC 0x80200000 /*Total cycles*/
#define PAPI_MIPS    0x80200001 /*Millions of instructions executed per second*/



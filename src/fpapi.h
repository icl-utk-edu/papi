#define PAPI_MAX_PRESET_EVENTS 64 

#define PAPI_L1_DCM  128 
#define PAPI_L1_ICM  129 
#define PAPI_L2_DCM  130 
#define PAPI_L2_ICM  131 
#define PAPI_L3_DCM  132 
#define PAPI_L3_ICM  133 
#define PAPI_L1_TCM  134 
#define PAPI_L2_TCM  135 
#define PAPI_L3_TCM  136 
#define PAPI_CA_SNP  137 
#define PAPI_CA_SHR  138 
#define PAPI_CA_CLN  139 
#define PAPI_CA_INV  140 
#define PAPI_CA_ITV  141 
#define PAPI_L3_LDM  142 
#define PAPI_L3_STM  143 
#define PAPI_BRU_IDL 144 
#define PAPI_FXU_IDL 145 
#define PAPI_FPU_IDL 146 
#define PAPI_LSU_IDL 147 
#define PAPI_TLB_DM  148 
#define PAPI_TLB_IM  149 
#define PAPI_TLB_TL  150 
#define PAPI_L1_LDM  151 
#define PAPI_L1_STM  152 
#define PAPI_L2_LDM  153 
#define PAPI_L2_STM  154 
#define PAPI_BTAC_M  155 
#define PAPI_PRF_DM  156 
#define PAPI_L3_DCH  157
#define PAPI_TLB_SD  158 
#define PAPI_CSR_FAL 159 
#define PAPI_CSR_SUC 160 
#define PAPI_CSR_TOT 161 
#define PAPI_MEM_SCY 162 
#define PAPI_MEM_RCY 163 
#define PAPI_MEM_WCY 164 
#define PAPI_STL_ICY 165 
#define PAPI_FUL_ICY 166 
#define PAPI_STL_CCY 167
#define PAPI_FUL_CCY 168 
#define PAPI_HW_INT  169 
#define PAPI_BR_UCN  170 
#define PAPI_BR_CN   171 
#define PAPI_BR_TKN  172 
#define PAPI_BR_NTK  173 
#define PAPI_BR_MSP  174 
#define PAPI_BR_PRC  175 
#define PAPI_FMA_INS 176 
#define PAPI_TOT_IIS 177 
#define PAPI_TOT_INS 178 
#define PAPI_INT_INS 179 
#define PAPI_FP_INS  180 
#define PAPI_LD_INS  181 
#define PAPI_SR_INS  182 
#define PAPI_BR_INS  183 
#define PAPI_VEC_INS 184 
#define PAPI_FLOPS   185 
#define PAPI_RES_STL 186 
#define PAPI_FP_STAL 187 
#define PAPI_TOT_CYC 188 
#define PAPI_IPS     189 
#define PAPI_LST_INS 190 
#define PAPI_SYC_INS 191 
#define PAPI_FORTRAN_MAX 192



#define PAPI_EINVAL   -1  
#define PAPI_ENOMEM   -2 
#define PAPI_ESYS     -3
#define PAPI_ESBSTR   -4  
#define PAPI_ECLOST   -5  
#define PAPI_EBUG     -6  
#define PAPI_ENOEVNT  -7  
#define PAPI_ECNFLCT  -8  
#define PAPI_ENOTRUN  -9 
#define PAPI_EMISC   -10

#define PAPI_NULL       -1 

#define PAPI_DOM_USER    1 
#define PAPI_DOM_MIN     PAPI_DOM_USER
#define PAPI_DOM_KERNEL	 2   
#define PAPI_DOM_OTHER	 4  
#define PAPI_DOM_ALL	 (PAPI_DOM_USER|PAPI_DOM_KERNEL|PAPI_DOM_OTHER)
#define PAPI_DOM_MAX     PAPI_DOM_ALL

/* Have to convert.... */
#define PAPI_DOM_HWSPEC  5 


#define PAPI_GRN_THR     1 
#define PAPI_GRN_MIN     PAPI_GRN_THR
#define PAPI_GRN_PROC    2
#define PAPI_GRN_PROCG   4 
#define PAPI_GRN_SYS     8 
#define PAPI_GRN_SYS_CPU 16
#define PAPI_GRN_MAX     PAPI_GRN_SYS_CPU

#define PAPI_PER_CPU     1  
#define PAPI_PER_NODE    2 
#define PAPI_SYSTEM	 3   

#define PAPI_PER_THR     0   
#define PAPI_PER_PROC    1  

#define PAPI_ONESHOT	 1 
#define PAPI_RANDOMIZE	 2
#define PAPI_DEF_MPXRES  1000 

#define PAPI_STOPPED      1 
#define PAPI_RUNNING      2 
#define PAPI_PAUSED       4
#define PAPI_NOT_INIT     8 
#define PAPI_OVERFLOWING  16 
#define PAPI_PROFILING    32 
#define PAPI_MULTIPLEXING 64 
#define PAPI_ACCUMULATING 128 

#define PAPI_NUM_ERRORS  11 
#define PAPI_QUIET       0 
#define PAPI_VERB_ECONT  1
#define PAPI_VERB_ESTOP  2  

#define PAPI_SET_MPXRES  1 
#define PAPI_GET_MPXRES  2
#define PAPI_DEBUG	 3
#define PAPI_SET_OVRFLO  4  
#define PAPI_GET_OVRFLO  5 
#define PAPI_SET_DEFDOM  6
#define PAPI_GET_DEFDOM  7  
#define PAPI_SET_DOMAIN  8  
#define PAPI_GET_DOMAIN  9 
#define PAPI_SET_DEFGRN  10   
#define PAPI_GET_DEFGRN  11  
#define PAPI_SET_GRANUL  12 
#define PAPI_GET_GRANUL  13
#define PAPI_SET_INHERIT 15
#define PAPI_GET_INHERIT 16
#define PAPI_INHERIT_ALL (pid_t)-1 
#define PAPI_INHERIT_NONE (pid_t)0 
#define PAPI_SET_BIND    17    
#define PAPI_GET_BIND    18  
#define PAPI_SET_THRID   19 
#define PAPI_GET_THRID   20   

#define PAPI_GET_CPUS    21    
#define PAPI_SET_CPUS    22    
#define PAPI_GET_THREADS 23    
#define PAPI_SET_THREADS 24    
#define PAPI_GET_NUMCTRS 25    
#define PAPI_SET_NUMCTRS 26    
#define PAPI_SET_PROFIL  27  
#define PAPI_GET_PROFIL  28 

#define PAPI_PROFIL_POSIX    0
#define PAPI_PROFIL_RANDOM   1 
#define PAPI_PROFIL_WEIGHTED 2 

#define PAPI_SET_ATTACH  29   
#define PAPI_GET_ATTACH  30  
#define PAPI_GET_PRELOAD 31 
#define PAPI_INIT_SLOTS  64  
#define PAPI_ERROR	 123  

#define PAPI_MAX_ERRMS   16   
#define PAPI_MAX_ERROR   10  
#define PAPI_GET_CLOCKRATE  	70 
#define PAPI_GET_HWINFO  	72 
#define PAPI_GET_EXEINFO  	73 
#define PAPI_MAX_STR_LEN        81
#define PAPI_DERIVED            1
#define PAPI_THREAD_CREATE      1
#define PAPI_THREAD_DESTROY     2


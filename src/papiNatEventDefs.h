
/* file: papiNatEventDefs.h

The following is a list of native hardware events defined for each platform.

  Created by Haiahng You ( you@cs.utk.edu )

*/

#ifdef _AIX
#ifdef _POWER4

#else
#define NATIVE_PM_CYC                     0|0<<8|0
#define NATIVE_PM_INST_CMPL               0|1<<8|0
#define NATIVE_PM_TB_BIT_TRANS            0|2<<8|0
#define NATIVE_PM_INST_DISP               0|3<<8|0
#define NATIVE_PM_LD_CMPL                 0|4<<8|0
#define NATIVE_PM_IC_MISS                 0|5<<8|0
#define NATIVE_PM_LD_MISS_L2HIT           0|6<<8|0
#define NATIVE_PM_LD_MISS_EXCEED_NO_L2    0|7<<8|0
#define NATIVE_PM_ST_MISS_EXCEED_NO_L2    0|9<<8|0
#define NATIVE_PM_BURSTRD_MISS_L2_INT     0|10<<8|0
#define NATIVE_PM_IC_MISS_USED            0|12<<8|0
#define NATIVE_PM_DU_ECAM_RCAM_OFFSET_HIT 0|13<<8|0
#define NATIVE_PM_GLOBAL_CANCEL_INST_DEL  0|14<<8|0
#define NATIVE_PM_CHAIN_1_TO_8	          0|15<<8|0
#define NATIVE_PM_FPU0_BUSY	              0|16<<8|0
#define NATIVE_PM_DSLB_MISS	              0|17<<8|0
#define NATIVE_PM_LSU0_ISS_TAG_ST	      0|18<<8|0
#define NATIVE_PM_TLB_MISS	              0|19<<8|0
#define NATIVE_PM_EE_OFF	              0|20<<8|0
#define NATIVE_PM_BRU_IDLE	              0|21<<8|0
#define NATIVE_PM_SYNCHRO_INST	          0|22<<8|0
#define NATIVE_PM_CYC_1STBUF_OCCP	      0|24<<8|0
#define NATIVE_PM_SNOOP_L1_M_TO_E_OR_S	  0|25<<8|0
#define NATIVE_PM_ST_CMPLBF_AT_GC	      0|26<<8|0
#define NATIVE_PM_LINK_STACK_FULL	      0|27<<8|0
#define NATIVE_PM_CBR_RESOLV_DISP	      0|28<<8|0
#define NATIVE_PM_LD_CMPLBF_AT_GC	      0|29<<8|0
#define NATIVE_PM_ENTRY_CMPLBF	          0|30<<8|0
#define NATIVE_PM_BIU_ST_RTRY	          0|32<<8|0
#define NATIVE_PM_EIEIO_WT_ST	          0|33<<8|0
#define NATIVE_PM_I_1_ST_TO_BUS	          0|35<<8|0
#define NATIVE_PM_CRB_BUSY_ENT	          0|36<<8|0
#define NATIVE_PM_DC_PREF_STREAM_ALLOC_BLK 0|37<<8|0
#define NATIVE_PM_W_1_ST	               0|38<<8|0
#define NATIVE_PM_LD_CI	                   0|39<<8|0
#define NATIVE_PM_4MISS	                   0|40<<8|0
#define NATIVE_PM_ST_GATH_BYTES	           0|41<<8|0
#define NATIVE_PM_DC_HIT_UNDER_MISS	       0|42<<8|0
#define NATIVE_PM_INTLEAVE_CONFL_STALLS	   0|43<<8|0
#define NATIVE_PM_DU1_REQ_ST_ADDR_XTION	   0|44<<8|0
#define NATIVE_PM_BTC_BTL_BLK	           0|45<<8|0
#define NATIVE_PM_FPU_SUCCESS_OOO_INST_SCHED 0|46<<8|0
#define NATIVE_PM_FPU_LD_ST_ISSUES	         0|47<<8|0
#define NATIVE_PM_FPU_FPSCR	                 0|48<<8|0
#define NATIVE_PM_FPU0_FSQRT	             0|49<<8|0
#define NATIVE_PM_FPU0_EXEC_ESTIMATE	     0|50<<8|0

#define NATIVE_PM_SNOOP_L2ACC		         0|4<<8|1
#define NATIVE_PM_DU0_REQ_ST_ADDR_XTION		 0|5<<8|1
#define NATIVE_PM_TAG_BURSTRD_L2MISS		 0|6<<8|1
#define NATIVE_PM_FPU_IQ_FULL		         0|7<<8|1
#define NATIVE_PM_BR_PRED		             0|8<<8|1
#define NATIVE_PM_ST_MISS_L1		         0|9<<8|1
#define NATIVE_PM_LD_MISS_EXCEED_L2		     0|10<<8|1
#define NATIVE_PM_L2ACC_BY_RWITM		     0|11<<8|1
#define NATIVE_PM_ST_MISS_EXCEED_L2		     0|12<<8|1
#define NATIVE_PM_ST_COND_FAIL		         0|13<<8|1
#define NATIVE_PM_CI_ST_WT_CI_ST		     0|14<<8|1
#define NATIVE_PM_CHAIN_2_TO_1		         0|15<<8|1
#define NATIVE_PM_TAG_BURSTRD_MISS_L2_INT	 0|16<<8|1
#define NATIVE_PM_FXU2_IDLE		             0|17<<8|1
#define NATIVE_PM_SC_INST		             0|18<<8|1
#define NATIVE_PM_2CASTOUT_BF		         0|20<<8|1
#define NATIVE_PM_BIU_LD_NORTRY		         0|21<<8|1
#define NATIVE_PM_RESRV_RQ		             0|22<<8|1
#define NATIVE_PM_SNOOP_E_TO_S		         0|23<<8|1
#define NATIVE_PM_IBUF_EMPTY		         0|25<<8|1
#define NATIVE_PM_SYNC_CMPLBF_CYC		     0|26<<8|1
#define NATIVE_PM_TLBSYNC_CMPLBF_CYC		 0|27<<8|1
#define NATIVE_PM_DC_PREF_L2_INV		     0|28<<8|1
#define NATIVE_PM_DC_PREF_FILT_1STR		     0|29<<8|1
#define NATIVE_PM_ST_CI_PREGATH		         0|30<<8|1
#define NATIVE_PM_ST_GATH_HW		         0|31<<8|1
#define NATIVE_PM_LD_WT_ADDR_CONF		     0|32<<8|1
#define NATIVE_PM_TAG_LD_DATA_RECV		     0|33<<8|1
#define NATIVE_PM_FPU1_DENORM		         0|34<<8|1
#define NATIVE_PM_FPU1_CMPL		             0|35<<8|1
#define NATIVE_PM_FPU_FEST		             0|36<<8|1
#define NATIVE_PM_FPU_LD		             0|37<<8|1
#define NATIVE_PM_FPU0_FDIV		             0|38<<8|1
#define NATIVE_PM_FPU0_FPSCR		         0|39<<8|1

#define NATIVE_PM_LD_MISS_L1			     0|5<<8|2
#define NATIVE_PM_TAG_ST_MISS_L2			 0|6<<8|2
#define NATIVE_PM_BRQ_FULL_CYC			     0|7<<8|2
#define NATIVE_PM_TAG_ST_MISS_L2_INT		 0|8<<8|2
#define NATIVE_PM_ST_CMPL			         0|9<<8|2
#define NATIVE_PM_TAG_ST_CMPL			     0|10<<8|2
#define NATIVE_PM_LD_NEXT			         0|11<<8|2
#define NATIVE_PM_ST_MISS_L2			     0|12<<8|2
#define NATIVE_PM_TAG_BURSTRD_L2ACC			 0|13<<8|2
#define NATIVE_PM_CHAIN_3_TO_2			     0|15<<8|2
#define NATIVE_PM_UNALIGNED_ST			     0|16<<8|2
#define NATIVE_PM_CORE_ST_N_COPYBACK		 0|17<<8|2
#define NATIVE_PM_SYNC_RERUN			     0|18<<8|2
#define NATIVE_PM_3CASTOUT_BF			     0|19<<8|2
#define NATIVE_PM_BIU_RETRY_DU_LOST_RES		 0|20<<8|2
#define NATIVE_PM_SNOOP_L2_E_OR_S_TO_I		 0|21<<8|2
#define NATIVE_PM_FPU_FDIV			         0|22<<8|2
#define NATIVE_PM_IO_INTERPT			     0|24<<8|2
#define NATIVE_PM_DC_PREF_HIT			     0|25<<8|2
#define NATIVE_PM_DC_PREF_FILT_2STR			 0|26<<8|2
#define NATIVE_PM_PREF_MATCH_DEM_MISS		 0|27<<8|2
#define NATIVE_PM_LSU1_IDLE			         0|28<<8|2

#define NATIVE_PM_FPU0_DENORM				 0|6<<8|3
#define NATIVE_PM_LSU0_ISS_TAG_LD			 0|7<<8|3
#define NATIVE_PM_TAG_ST_L2ACC				 0|8<<8|3
#define NATIVE_PM_LSU0_LD_DATA				 0|9<<8|3
#define NATIVE_PM_ST_MISS_L2_INT			 0|10<<8|3
#define NATIVE_PM_SYNC				         0|11<<8|3
#define NATIVE_PM_FXU2_BUSY				     0|13<<8|3
#define NATIVE_PM_BIU_ST_NORTRY				 0|14<<8|3
#define NATIVE_PM_CHAIN_4_TO_3				 0|15<<8|3
#define NATIVE_PM_DC_ALIAS_HIT				 0|16<<8|3
#define NATIVE_PM_FXU1_IDLE				     0|17<<8|3
#define NATIVE_PM_UNALIGNED_LD				 0|18<<8|3
#define NATIVE_PM_CMPLU_WT_LD				 0|19<<8|3
#define NATIVE_PM_BIU_ARI_RTRY				 0|20<<8|3
#define NATIVE_PM_FPU_FSQRT				     0|21<<8|3
#define NATIVE_PM_BR_CMPL				     0|22<<8|3
#define NATIVE_PM_DISP_BF_EMPTY				 0|23<<8|3
#define NATIVE_PM_LNK_REG_STACK_ERR			 0|24<<8|3
#define NATIVE_PM_CRLU_PROD_RES				 0|25<<8|3
#define NATIVE_PM_TLBSYNC_RERUN				 0|26<<8|3
#define NATIVE_PM_SNOOP_L2_M_TO_E_OR_S		 0|27<<8|3
#define NATIVE_PM_DEM_FETCH_WT_PREF			 0|29<<8|3
#define NATIVE_PM_FPU0_FRSP_FCONV			 0|30<<8|3

#define NATIVE_PM_IC_HIT					 0|1<<8|4
#define NATIVE_PM_0INST_CMPL				 0|2<<8|4
#define NATIVE_PM_FPU_DENORM				 0|3<<8|4
#define NATIVE_PM_BURSTRD_L2ACC				 0|4<<8|4
#define NATIVE_PM_FPU0_CMPL					 0|5<<8|4
#define NATIVE_PM_LSU_IDLE					 0|6<<8|4
#define NATIVE_PM_BTAC_HITS					 0|7<<8|4
#define NATIVE_PM_STQ_FULL					 0|8<<8|4
#define NATIVE_PM_BIU_WT_ST_BF				 0|9<<8|4
#define NATIVE_PM_SNOOP_L2_M_TO_I			 0|10<<8|4
#define NATIVE_PM_FPU_FRSP_FCONV			 0|11<<8|4
#define NATIVE_PM_BIU_ASI_RTRY				 0|13<<8|4
#define NATIVE_PM_CHAIN_5_TO_4				 0|15<<8|4
#define NATIVE_PM_DC_REQ_HIT_PREF_BUF		 0|16<<8|4
#define NATIVE_PM_DC_PREF_FILT_3STR			 0|17<<8|4
#define NATIVE_PM_3MISS					     0|18<<8|4
#define NATIVE_PM_ST_GATH_WORD			     0|19<<8|4
#define NATIVE_PM_LD_WT_ST_CONF				 0|20<<8|4
#define NATIVE_PM_LSU1_ISS_TAG_ST			 0|21<<8|4
#define NATIVE_PM_FPU1_BUSY					 0|22<<8|4
#define NATIVE_PM_FPU0_FMOV_FEST			 0|23<<8|4
#define NATIVE_PM_4CASTOUT_BUF				 0|24<<8|4

#define NATIVE_PM_ST_HIT_L1					0|1<<8|5
#define NATIVE_PM_FXU2_PROD_RESULT			0|2<<8|5
#define NATIVE_PM_BTAC_MISS					0|3<<8|5
#define NATIVE_PM_CBR_DISP					0|5<<8|5
#define NATIVE_PM_LQ_FULL					0|6<<8|5
#define NATIVE_PM_6XXBUS_CMPL_LOAD			0|7<<8|5
#define NATIVE_PM_SNOOP_PUSH_INT			0|8<<8|5
#define NATIVE_PM_EE_OFF_EXT_INT			0|9<<8|5
#define NATIVE_PM_BIU_LD_RTRY				0|10<<8|5
#define NATIVE_PM_FPU_EXE_FCMP				0|11<<8|5
#define NATIVE_PM_DC_PREF_BF_INV			0|13<<8|5
#define NATIVE_PM_DC_PREF_FILT_4STR			0|14<<8|5
#define NATIVE_PM_CHAIN_6_TO_5				0|15<<8|5
#define NATIVE_PM_1MISS						0|16<<8|5
#define NATIVE_PM_ST_GATH_DW				0|17<<8|5
#define NATIVE_PM_LSU1_ISS_TAG_LD			0|18<<8|5
#define NATIVE_PM_FPU1_IDLE					0|19<<8|5
#define NATIVE_PM_FPU0_FMA					0|20<<8|5
#define NATIVE_PM_SNOOP_PUSH_BUF			0|21<<8|5

#define NATIVE_PM_FXU0_PROD_RESULT			0|1<<8|6
#define NATIVE_PM_BR_DISP					0|2<<8|6
#define NATIVE_PM_MPRED_BR_CAUSED_GC		0|3<<8|6
#define NATIVE_PM_SNOOP						0|4<<8|6
#define NATIVE_PM_0INST_DISP				0|6<<8|6
#define NATIVE_PM_FXU_IDLE					0|7<<8|6
#define NATIVE_PM_6XX_RTRY_CHNG_TRTP		0|8<<8|6
#define NATIVE_PM_FPU_FMA					0|9<<8|6
#define NATIVE_PM_ST_DISP					0|10<<8|6
#define NATIVE_PM_DC_PREF_L2HIT				0|14<<8|6
#define NATIVE_PM_CHAIN_7_TO_6				0|15<<8|6
#define NATIVE_PM_DC_PREF_BLOCK_DEMAND_MISS	0|16<<8|6
#define NATIVE_PM_2MISS						0|17<<8|6
#define NATIVE_PM_DC_PREF_USED				0|18<<8|6
#define NATIVE_PM_LSU_WT_SNOOP_BUSY			0|19<<8|6
#define NATIVE_PM_IC_PREF_USED				0|20<<8|6
#define NATIVE_PM_FPU0_FADD_FCMP_FMUL		0|22<<8|6
#define NATIVE_PM_1WT_THRU_BUF_USED			0|23<<8|6

#define NATIVE_PM_SNOOP_L2HIT				0|1<<8|7
#define NATIVE_PM_BURSTRD_L2MISS			0|2<<8|7
#define NATIVE_PM_RESRV_CMPL				0|3<<8|7
#define NATIVE_PM_FXU1_PROD_RESULT			0|4<<8|7
#define NATIVE_PM_RETRY_BUS_OP				0|5<<8|7
#define NATIVE_PM_FPU_IDLE					0|6<<8|7
#define NATIVE_PM_FETCH_CORR_AT_DISPATCH	0|7<<8|7
#define NATIVE_PM_CMPLU_WT_ST				0|8<<8|7
#define NATIVE_PM_FPU_FADD_FMUL				0|9<<8|7
#define NATIVE_PM_LD_DISP					0|10<<8|7
#define NATIVE_PM_ALIGN_INT					0|11<<8|7
#define NATIVE_PM_2WT_THRU_BUF_USED			0|14<<8|7
#define NATIVE_PM_CHAIN_8_TO_7				0|15<<8|7

#endif
#endif
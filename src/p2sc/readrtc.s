#  COMPONENT_NAME: non-system routine
#
#  ORIGIN: IBM
#
#  IBM CONFIDENTIAL
#  Copyright International Business Machines Corp. 1988
#  Unpublished Work
#  All Rights Reserved
#  Licensed Material - Property of IBM
#
#  RESTRICTED RIGHTS LEGEND
#  Use, Duplication or Disclosure by the Government is subject to
#  restrictions as set forth in paragraph (b)(3)(B) of the Rights in
#  Technical Data and Computer Software clause in DAR 7-104.9(a).
#
 
#
#  NAME: times
#
#  FUNCTION: Read Realtime clock
#
#  EXECUTION ENVIRONMENT:
#  Standard register usage and linkage convention.
#  Registers used r3-r8
#  Condition registers used: 0
#  No stack requirements.
#  No TOC requirements.
#
#  NOTES:
#  The read is tested to insure a valid value as per the architecture document.
#  Temporary substitute for system times function.
#
#  The addresses are treated as usigned quantities,
#
#  RETURN VALUE DESCRIPTION: Value of RTCU to 0(3), value of RTCL to 4(3).
#
#  r3=0 if ok   r3=-1 if no good read
#
 
#          S_PROLOG( readrtc )
#
#  Calling sequence:
#       R3   Address of 2 word block for RTC results
#
          .toc
	  .csect readrtc[DS]
          .globl readrtc[DS]
 
          .long  .readrtc[PR]
          .long  TOC[t0]
     
.readrtc:
          .csect .readrtc[PR]
          .globl .readrtc[PR]
            
          mfspr    5,4               # RTCU
          mfspr    6,5               # RTCL
          mfspr    7,4               # RTCU
          mr       4,3               # Save address in case of retry
          cmpl     0,5,7             # Test for valid read
          lil      3,0               # Zero return code and data
          st       5,0(4)            # Return RTCU
          st       6,4(4)            # Return RTCL
          beqr                       # Return if valid read - r3 = 0
          lil      8,32              # Retry count - safety valve
          mtctr    8                 # Count reg has retry count
ReRead:
          mfspr    5,4               # RTCU
          mfspr    6,5               # RTCL
          mfspr    7,4               # RTCU
          cmpl     0,5,7             # Test for valid read
          st       5,0(4)            # Return RTCU
          st       6,8(4)            # Return RTCL
          beqr                       # Return if valid read - r3 = 0
          bc       16,0,ReRead       # Dec count and continue(branch) if not 0
          lil      3,-1              # Bad return code - retry count exceeded
          br                         # Return with failure code - r3 ^= 0
          .align  2
          .byte   0xdf,2,0xdf,0


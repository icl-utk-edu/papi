;===============================================================

;The system calls:

;=================================================================

;
;       (C) COPYRIGHT CRAY RESEARCH, INC.
;       UNPUBLISHED PROPRIETARY INFORMATION.
;       ALL RIGHTS RESERVED.
;

        .ident  pmctr

#include <mpp/regdef.h>
#include <mpp/pal.h>
#include <mpp/syscall.h>


        .psect  pmctr@code,code

perfmonctl::
        lda     v0, 278
        call_pal        136                     ; call unicos
        ret     zero, (ra)

_wrperf::
        call_pal        141
        ret     zero, (ra)

_rdperf::
        bis     a0, a0, t1
        call_pal        140
        stq     v0, 0(t1)
        stq     a0, 8(t1)
        stq     a1, 16(t1)
        stq     a2, 24(t1)

        bis     zero, zero, v0
        ret     zero, (ra)

        .endp

        .psect  usmid,data
        .asciz  "@(#)pat/lib/cal.s      10.1    01/23/97 10:10:08"

        .end



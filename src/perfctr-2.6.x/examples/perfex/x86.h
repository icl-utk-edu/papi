/* $Id: x86.h,v 1.1.2.1 2010/06/08 20:48:56 mikpe Exp $
 * x86-specific declarations.
 *
 * Copyright (C) 1999-2010  Mikael Pettersson
 */

#define ARCH_LONG_OPTIONS	\
    { "nhlm_offcore_rsp_0", 1, NULL, 1 }, \
    { "nhlm_offcore_rsp_1", 1, NULL, 2 }, \
    { "p4pe", 1, NULL, 1 }, { "p4_pebs_enable", 1, NULL, 1 }, \
    { "p4pmv", 1, NULL, 2 }, { "p4_pebs_matrix_vert", 1, NULL, 2 },

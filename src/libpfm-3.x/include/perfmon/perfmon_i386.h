/*
 * Copyright (c) 2005-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * This file contains X86  Processor Family (32-bit and 64-bit) specific
 * definitions for the perfmon interface.It should never be included directly, use
 * <perfmon/perfmon.h> instead.
 */
#ifndef _PERFMON_I386_H_
#define _PERFMON_I386_H_

/*
 * X86 specific context flags
 */
#define PFM_X86_FL_INSECURE  0x10000         /* allow rdpmc at user level */

#endif /* _PERFMON_I386_H_ */

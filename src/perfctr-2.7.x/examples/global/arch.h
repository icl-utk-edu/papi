/* $Id$
 * Architecture-specific support code.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */

extern int counting_mips;	/* for CPUs that cannot FLOPS */

extern void setup_control(const struct perfctr_info *info,
			  struct perfctr_cpu_control *cpu_control);

/* $Id$
 * Architecture-specific support code.
 *
 * Copyright (C) 2004  Mikael Pettersson
 */

extern unsigned long mcontext_pc(const mcontext_t *mc);

extern void do_setup(const struct perfctr_info *info,
		     struct perfctr_cpu_control *cpu_control);

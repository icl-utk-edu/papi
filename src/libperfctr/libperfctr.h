/* $Id$
 * Library interface to Linux x86 Performance-Monitoring Counters.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */

#ifndef __LIB_PERFCTR_H
#define __LIB_PERFCTR_H

#include "linux/perfctr.h"

/*
 * Invokes the kernel driver directly.
 */

extern int perfctr_syscall(unsigned cmd, long arg);

/*
 * Operations on explicitly attached virtual-mode perfctrs.
 */

struct vperfctr;	/* opaque; users only see pointers to these */

extern struct vperfctr *perfctr_attach_rdonly(int pid);
extern struct vperfctr *perfctr_attach_rdwr(int pid, struct perfctr_control*);
extern int perfctr_read(const struct vperfctr*, struct vperfctr_state*);
extern int perfctr_control(const struct vperfctr*, struct perfctr_control*);
extern int perfctr_stop(const struct vperfctr*);
extern int perfctr_unlink(const struct vperfctr*);
extern void perfctr_close(struct vperfctr*);

/*
 * Operations on the process' own virtual-mode perfctrs.
 */

extern struct vperfctr *perfctr_attach_rdonly_self(void);
extern struct vperfctr *perfctr_attach_rdwr_self(void);
extern int perfctr_read_self(const struct vperfctr*, struct vperfctr_state*);
extern int perfctr_control_self(const struct vperfctr*, struct perfctr_control*);
extern int perfctr_stop_self(const struct vperfctr*);
extern int perfctr_unlink_self(void);
extern void perfctr_close_self(struct vperfctr*);

/*
 * Operations on global-mode perfctrs.
 */

extern int perfctr_global_control(struct gperfctr_control*);
extern int perfctr_global_read(struct gperfctr_state*);
extern int perfctr_global_stop(void);

/*
 * Miscellaneous operations.
 */

extern int perfctr_info(struct perfctr_info*);
extern unsigned perfctr_cpu_nrctrs(void);
extern const char *perfctr_cpu_name(void);
extern unsigned perfctr_evntsel_num_insns(void);

#endif /* __LIB_PERFCTR_H */

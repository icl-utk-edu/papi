/* $Id$
 * Library interface to Linux x86 Performance-Monitoring Counters.
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */

#ifndef __LIB_PERFCTR_H
#define __LIB_PERFCTR_H

#include "linux/perfctr.h"

/*
 * Raw device interface.
 */

struct perfctr_dev;	/* opaque */
struct perfctr_dev *perfctr_dev_open(void);
void perfctr_dev_close(struct perfctr_dev*);
int perfctr_syscall(const struct perfctr_dev *dev, unsigned cmd, long arg);

/*
 * Operations on the process' own virtual-mode perfctrs.
 */

struct vperfctr;	/* opaque */

/* Create an access point to your own vperfctr. 'dev' can be closed
   afterwards: the vperfctr remains valid until closed. */
struct vperfctr *vperfctr_attach(const struct perfctr_dev *dev);

unsigned long long vperfctr_read_one(const struct vperfctr*, int);
int vperfctr_read_state(const struct vperfctr*, struct vperfctr_state*);
int vperfctr_control(const struct vperfctr*, struct perfctr_control*);
int vperfctr_stop(const struct vperfctr*);
int vperfctr_unlink(const struct vperfctr*);
void vperfctr_close(struct vperfctr*);

/*
 * Operations on global-mode perfctrs.
 */

int perfctr_global_control(const struct perfctr_dev*, struct gperfctr_control*);
int perfctr_global_read(const struct perfctr_dev*, struct gperfctr_state*);
int perfctr_global_stop(const struct perfctr_dev*);

/*
 * Miscellaneous operations.
 */

int perfctr_info(const struct perfctr_dev*, struct perfctr_info*);
unsigned perfctr_cpu_nrctrs(const struct perfctr_dev*);
const char *perfctr_cpu_name(const struct perfctr_dev*);
unsigned perfctr_evntsel_num_insns(const struct perfctr_dev*);

#endif /* __LIB_PERFCTR_H */

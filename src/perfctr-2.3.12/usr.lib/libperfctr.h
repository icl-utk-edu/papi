/* $Id$
 * Library interface to Linux x86 Performance-Monitoring Counters.
 *
 * Copyright (C) 1999-2002  Mikael Pettersson
 */

#ifndef __LIB_PERFCTR_H
#define __LIB_PERFCTR_H

#include "linux/perfctr.h"

/*
 * Operations on the process' own virtual-mode perfctrs.
 */

struct vperfctr;	/* opaque */

struct vperfctr *vperfctr_open(void);
int vperfctr_info(const struct vperfctr*, struct perfctr_info*);
unsigned long long vperfctr_read_tsc(const struct vperfctr*);
unsigned long long vperfctr_read_pmc(const struct vperfctr*, unsigned);
void vperfctr_read_ctrs(const struct vperfctr*, struct perfctr_sum_ctrs*);
int vperfctr_read_state(const struct vperfctr*, struct perfctr_sum_ctrs*,
			struct vperfctr_control*);
int vperfctr_control(const struct vperfctr*, struct vperfctr_control*);
int vperfctr_stop(const struct vperfctr*);
int vperfctr_unlink(const struct vperfctr*);
void vperfctr_close(struct vperfctr*);

/*
 * Operations on global-mode perfctrs.
 */

struct gperfctr;	/* opaque */

struct gperfctr *gperfctr_open(void);
void gperfctr_close(struct gperfctr*);
int gperfctr_control(const struct gperfctr*, struct gperfctr_control*);
int gperfctr_read(const struct gperfctr*, struct gperfctr_state*);
int gperfctr_stop(const struct gperfctr*);
int gperfctr_info(const struct gperfctr*, struct perfctr_info*);

/*
 * Descriptions of the events available for different processor types.
 */

struct perfctr_event {
    const char *name;
    unsigned int code;
    unsigned int counters_mask; /* (1<<i) is set if ctr(i) is permitted */
    unsigned int default_qualifier;
};

struct perfctr_event_set {
    unsigned int cpu_type;
    const struct perfctr_event_set *include;
    unsigned int nevents;
    const struct perfctr_event *events;
};

extern const struct perfctr_event_set perfctr_generic_event_set;
extern const struct perfctr_event_set perfctr_p5_event_set;
extern const struct perfctr_event_set perfctr_p5mmx_event_set;
extern const struct perfctr_event_set perfctr_mii_event_set;
extern const struct perfctr_event_set perfctr_wcc6_event_set;
extern const struct perfctr_event_set perfctr_wc2_event_set;
extern const struct perfctr_event_set perfctr_p6_event_set;
extern const struct perfctr_event_set perfctr_pii_event_set;
extern const struct perfctr_event_set perfctr_piii_event_set;
extern const struct perfctr_event_set perfctr_vc3_event_set;
extern const struct perfctr_event_set perfctr_k7_event_set;

const struct perfctr_event_set *perfctr_cpu_event_set(unsigned int cpu_type);

/*
 * Miscellaneous operations.
 */

unsigned perfctr_cpu_nrctrs(const struct perfctr_info*);
const char *perfctr_cpu_name(const struct perfctr_info*);
void perfctr_print_info(const struct perfctr_info*);

#endif /* __LIB_PERFCTR_H */

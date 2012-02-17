/* $Id: event_set.h,v 1.5 2004/02/20 21:32:06 mikpe Exp $
 * Common definitions used when creating event set descriptions.
 *
 * Copyright (C) 2003-2004  Mikael Pettersson
 */
#define ARRAY_SIZE(x)	(sizeof(x) / sizeof((x)[0]))

#define UM(um)	((const struct perfctr_unit_mask*)&(um).header)

struct perfctr_unit_mask_header {
    unsigned short default_value;
    enum perfctr_unit_mask_type type:8;
    unsigned char nvalues;
};

struct perfctr_unit_mask_0 {
    struct perfctr_unit_mask_header header;
};

struct perfctr_unit_mask_1 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[1];
};

struct perfctr_unit_mask_2 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[2];
};

struct perfctr_unit_mask_3 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[3];
};

struct perfctr_unit_mask_4 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[4];
};

struct perfctr_unit_mask_5 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[5];
};

struct perfctr_unit_mask_6 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[6];
};

struct perfctr_unit_mask_7 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[7];
};

struct perfctr_unit_mask_8 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[8];
};

struct perfctr_unit_mask_9 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[9];
};

struct perfctr_unit_mask_13 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[13];
};

struct perfctr_unit_mask_15 {
    struct perfctr_unit_mask_header header;
    struct perfctr_unit_mask_value values[15];
};

extern const struct perfctr_event_set perfctr_p5_event_set;
extern const struct perfctr_event_set perfctr_p5mmx_event_set;
extern const struct perfctr_event_set perfctr_mii_event_set;
extern const struct perfctr_event_set perfctr_wcc6_event_set;
extern const struct perfctr_event_set perfctr_wc2_event_set;
extern const struct perfctr_event_set perfctr_vc3_event_set;
extern const struct perfctr_event_set perfctr_ppro_event_set;
extern const struct perfctr_event_set perfctr_p2_event_set;
extern const struct perfctr_event_set perfctr_p3_event_set;
extern const struct perfctr_event_set perfctr_p4_event_set;
extern const struct perfctr_event_set perfctr_k7_event_set;
extern const struct perfctr_event_set perfctr_k8_event_set;
extern const struct perfctr_event_set perfctr_pentm_event_set;
extern const struct perfctr_event_set perfctr_k8c_event_set;
extern const struct perfctr_event_set perfctr_p4m3_event_set;

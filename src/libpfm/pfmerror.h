/*
 * pferror.h
 *
 * Copyright (C) 2001 Hewlett-Packard Co
 * Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com>
 */
#ifndef __PFMERROR_H__
#define __PFMERROR_H__

#define PFME_BASE		-100
#define PFME_UNKNOWN_EVENT	PFME_BASE
#define PFME_INVAL		(PFME_BASE+1)

extern const char *pfm_strerror(int error);

#endif /* __PFMERROR_H__ */

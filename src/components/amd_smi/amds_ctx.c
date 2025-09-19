/**
 * @file    amds_ctx.c
 * @author  Dong Jun Woun 
 *          djwoun@gmail.com
 *
 */

#include "amds.h"
#include "amds_priv.h"
#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include <stdint.h>   // for uint64_t, INT types

unsigned int _amd_smi_lock;

/* Use a 64-bit global device mask to support up to 64 devices */
static uint64_t device_mask = 0;

static int acquire_devices(unsigned int *events_id, int num_events, uint64_t *bitmask) {
  if (!bitmask) return PAPI_EINVAL;
  if (num_events < 0) return PAPI_EINVAL;
  if (num_events > 0 && !events_id) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  uint64_t mask_acq = 0;
  for (int i = 0; i < num_events; ++i) {
    unsigned int id = events_id[i];
    if (id >= (unsigned)ntv_table_p->count) {
      return PAPI_EINVAL;
    }
    int dev_id = ntv_table_p->events[id].device;
    if (dev_id < 0) continue;                 /* no device associated */
    if (dev_id >= 64) return PAPI_EINVAL;     /* out of representable range */
    mask_acq |= (UINT64_C(1) << dev_id);
  }

  _papi_hwi_lock(_amd_smi_lock);
  if (mask_acq & device_mask) {
    _papi_hwi_unlock(_amd_smi_lock);
    return PAPI_ECNFLCT; // conflict: device already in use
  }
  device_mask |= mask_acq;
  _papi_hwi_unlock(_amd_smi_lock);

  *bitmask = mask_acq;
  return PAPI_OK;
}

static int release_devices(uint64_t *bitmask) {
  if (!bitmask) return PAPI_EINVAL;
  uint64_t mask_rel = *bitmask;

  _papi_hwi_lock(_amd_smi_lock);
  if ((mask_rel & device_mask) != mask_rel) {
    _papi_hwi_unlock(_amd_smi_lock);
    return PAPI_EMISC;
  }
  /* Clear with &= ~mask for clarity and robustness */
  device_mask &= ~mask_rel;
  _papi_hwi_unlock(_amd_smi_lock);

  *bitmask = 0;
  return PAPI_OK;
}

/* Context management: open/close, start/stop, read/write, reset */
struct amds_ctx {
  int state;
  unsigned int *events_id;
  int num_events;
  long long *counters;
  uint64_t device_mask;       /* was int32_t: now 64-bit to match global */
};

int amds_ctx_open(unsigned int *event_ids, int num_events, amds_ctx_t *ctx) {
  if (!ctx) return PAPI_EINVAL;
  if (num_events < 0) return PAPI_EINVAL;
  if (num_events > 0 && !event_ids) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  amds_ctx_t new_ctx = (amds_ctx_t)papi_calloc(1, sizeof(struct amds_ctx));
  if (new_ctx == NULL) {
    return PAPI_ENOMEM;
  }
  new_ctx->events_id = event_ids; // Store pointer (caller must keep stable)
  new_ctx->num_events = num_events;
  new_ctx->counters = (long long *)papi_calloc((size_t)num_events, sizeof(long long));
  if (new_ctx->counters == NULL) {
    papi_free(new_ctx);
    return PAPI_ENOMEM;
  }

  /* Validate event ids range before acquiring devices */
  for (int i = 0; i < num_events; ++i) {
    if (event_ids[i] >= (unsigned)ntv_table_p->count) {
      papi_free(new_ctx->counters);
      papi_free(new_ctx);
      return PAPI_EINVAL;
    }
  }

  /* Acquire devices needed by these events to avoid conflicts */
  int papi_errno = acquire_devices(event_ids, num_events, &new_ctx->device_mask);
  if (papi_errno != PAPI_OK) {
    papi_free(new_ctx->counters);
    papi_free(new_ctx);
    return papi_errno;
  }

  for (int i = 0; i < num_events; ++i) {
    native_event_t *ev = &ntv_table_p->events[event_ids[i]];
    if (ev->open_func) {
      papi_errno = ev->open_func(ev);
      if (papi_errno != PAPI_OK) {
        for (int j = 0; j < i; ++j) {
          native_event_t *prev = &ntv_table_p->events[event_ids[j]];
          if (prev->close_func)
            prev->close_func(prev);
        }
        release_devices(&new_ctx->device_mask);
        papi_free(new_ctx->counters);
        papi_free(new_ctx);
        return papi_errno;
      }
    }
  }
  *ctx = new_ctx;
  return PAPI_OK;
}

int amds_ctx_close(amds_ctx_t ctx) {
  if (!ctx)
    return PAPI_OK;
  if (!ntv_table_p) {
    /* Best effort: release devices and free even if table is gone */
    (void)release_devices(&ctx->device_mask);
    papi_free(ctx->counters);
    papi_free(ctx);
    return PAPI_OK;
  }
  for (int i = 0; i < ctx->num_events; ++i) {
    unsigned int id = ctx->events_id[i];
    if (id >= (unsigned)ntv_table_p->count) continue;
    native_event_t *ev = &ntv_table_p->events[id];
    if (ev->close_func)
      ev->close_func(ev);
  }
  /* release device usage */
  (void)release_devices(&ctx->device_mask);
  papi_free(ctx->counters);
  papi_free(ctx);
  return PAPI_OK;
}

int amds_ctx_start(amds_ctx_t ctx) {
  if (!ctx) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  int papi_errno = PAPI_OK;
  for (int i = 0; i < ctx->num_events; ++i) {
    unsigned int id = ctx->events_id[i];
    if (id >= (unsigned)ntv_table_p->count) return PAPI_EINVAL;
    native_event_t *ev = &ntv_table_p->events[id];
    if (ev->start_func) {
      papi_errno = ev->start_func(ev);
      if (papi_errno != PAPI_OK)
        return papi_errno;
    }
  }
  ctx->state |= AMDS_EVENTS_RUNNING;
  return papi_errno;
}

int amds_ctx_stop(amds_ctx_t ctx) {
  if (!ctx) return PAPI_EINVAL;
  if (!(ctx->state & AMDS_EVENTS_RUNNING)) {
    return PAPI_OK;
  }
  if (!ntv_table_p) return PAPI_ECMP;

  int papi_errno = PAPI_OK;
  for (int i = 0; i < ctx->num_events; ++i) {
    unsigned int id = ctx->events_id[i];
    if (id >= (unsigned)ntv_table_p->count) continue;
    native_event_t *ev = &ntv_table_p->events[id];
    if (ev->stop_func) {
      int papi_errno_stop = ev->stop_func(ev);
      if (papi_errno == PAPI_OK)
        papi_errno = papi_errno_stop;
    }
  }
  ctx->state &= ~AMDS_EVENTS_RUNNING;
  return papi_errno;
}

int amds_ctx_read(amds_ctx_t ctx, long long **counts) {
  if (!ctx || !counts) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  /* Always produce a fully defined buffer */
  for (int i = 0; i < ctx->num_events; ++i) {
    ctx->counters[i] = 0;  /* default if read fails */
  }

  /* Optional: track first error, but don't bail early */
  int papi_errno = PAPI_OK;

  for (int i = 0; i < ctx->num_events; ++i) {
    unsigned int id = ctx->events_id[i];
    if (id >= (unsigned)ntv_table_p->count) {
      if (papi_errno == PAPI_OK) papi_errno = PAPI_EINVAL;
      continue;
    }
    native_event_t *ev = &ntv_table_p->events[id];

    int papi_errno_access = PAPI_OK;
    if (ev->access_func) {
      papi_errno_access = ev->access_func(PAPI_MODE_READ, ev);
    } else {
      papi_errno_access = PAPI_ECMP;
    }

    if (papi_errno_access == PAPI_OK) {
      ctx->counters[i] = (long long)ev->value;
    } else if (papi_errno == PAPI_OK) {
      papi_errno = papi_errno_access;  /* remember, but keep going */
    }
  }

  *counts = ctx->counters;
  
  return papi_errno;
}

int amds_ctx_write(amds_ctx_t ctx, long long *counts) {
  if (!ctx || !counts) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  int papi_errno = PAPI_OK;
  for (int i = 0; i < ctx->num_events; ++i) {
    unsigned int id = ctx->events_id[i];
    if (id >= (unsigned)ntv_table_p->count) return PAPI_EINVAL;
    native_event_t *ev = &ntv_table_p->events[id];
    if (!ev->access_func) return PAPI_ECMP;
    ev->value = counts[i];
    papi_errno = ev->access_func(PAPI_MODE_WRITE, ev);
    if (papi_errno != PAPI_OK) {
      return papi_errno;
    }
  }
  return papi_errno;
}

int amds_ctx_reset(amds_ctx_t ctx) {
  if (!ctx) return PAPI_EINVAL;
  if (!ntv_table_p) return PAPI_ECMP;

  for (int i = 0; i < ctx->num_events; ++i) {
    unsigned int id = ctx->events_id[i];
    if (id >= (unsigned)ntv_table_p->count) return PAPI_EINVAL;
    native_event_t *ev = &ntv_table_p->events[id];
    ev->value = 0;
    if (ctx->counters) ctx->counters[i] = 0;
  }
  return PAPI_OK;
}

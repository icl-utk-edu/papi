#ifndef _FLOPS_
#define _FLOPS_

#include "hw_desc.h"

void exec_flops(int double_precision, int EventSet, int retval, FILE *fp);
void flops_driver(char* papi_event_str, hw_desc_t *hw_desc, char* outdir);

#endif

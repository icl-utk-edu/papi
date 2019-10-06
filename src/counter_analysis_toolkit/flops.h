#ifndef _FLOPS_
#define _FLOPS_

void exec_flops(int double_precision, int EventSet, int retval, FILE *fp);
void flops_driver(char* papi_event_str, char* outdir);

#endif

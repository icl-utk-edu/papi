/*
 * pfmlib_os.c: set of functions OS dependent functions
 *
 * Copyright (c) 2003-2006 Hewlett-Packard Development Company, L.P.
 * Contributed by Stephane Eranian <eranian@hpl.hp.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <windows.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* for getline */
#endif
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <perfmon/perfmon.h>

#include <perfmon/pfmlib.h>
#include "../lib/pfmlib_priv.h"
#include "WinPMC.h"
/*
 * helper function to retrieve one value from /proc/cpuinfo
 * for internal libpfm use only
 * attr: the attribute (line) to look for
 * ret_buf: a buffer to store the value of the attribute (as a string)
 * maxlen : number of bytes of capacity in ret_buf
 *
 * ret_buf is null terminated.
 *
 * Return:
 * 	0 : attribute found, ret_buf populated
 * 	-1: attribute not found
 */
int
__pfm_getcpuinfo_attr(const char *attr, char *ret_buf, size_t maxlen)
{
  uint32_t regs[4];
  if (!strcmp(attr, "vendor_id"))
  {
	__cpuid(regs, 0);
	((uint32_t*)ret_buf)[0] = regs[1];
	((uint32_t*)ret_buf)[1] = regs[3];
	((uint32_t*)ret_buf)[2] = regs[2];
	ret_buf[12] = '\0';
	return 0;
  }
  else if (!strcmp(attr, "model"))
  {
	__cpuid(regs, 1);
	sprintf(ret_buf, "%d", (regs[0] >> 4) & 0xf);
	return 0;
  }
  else if (!strcmp(attr, "cpu family"))
  {
	__cpuid(regs, 1);
	sprintf(ret_buf, "%d", (regs[0] >> 8) & 0xf);
	return 0;
  }
  return -1;
}

#define PMC_DEVICE "\\\\.\\WinPMC"

HANDLE g_pmc_driver;

void OpenWinPMCDriver()
{
	HANDLE	process;
	DWORD_PTR processAffinityMask, systemAffinityMask;

  if (g_pmc_driver)
    return;

	/* force this process to run only on the lowest available processor */
	process = GetCurrentProcess();
	if (GetProcessAffinityMask(process, &processAffinityMask, &systemAffinityMask)) {
		/* set the mask to the lowest possible processor
			(I think this one should ALWAYS be here...) */
		processAffinityMask = 0x00000001;
		/* scan for the lowest bit in the system mask */
		while (!processAffinityMask & systemAffinityMask)
			processAffinityMask <<= 1;
		/* set affinity to lowest processor only */
		if (!SetProcessAffinityMask(process, processAffinityMask)) {
      return;
		}
	}
	else
    return;

	g_pmc_driver = CreateFile(PMC_DEVICE, GENERIC_READ | GENERIC_WRITE, 
     FILE_SHARE_READ | FILE_SHARE_WRITE, 0, OPEN_EXISTING, 0, 0);
}

void CloseWinPMCDriver()
{
	HANDLE process;
	DWORD_PTR processAffinityMask, systemAffinityMask;

  if (g_pmc_driver > 0 ) CloseHandle(g_pmc_driver);
    g_pmc_driver = 0;

	/* restore this process to run on all available processors */
	process = GetCurrentProcess();
	if (GetProcessAffinityMask(process, &processAffinityMask, &systemAffinityMask)) {
		/* set affinity to all processors */
		if (!SetProcessAffinityMask(process, systemAffinityMask)) {
			return;
		}
	}
}

int
pfm_create_context(pfarg_ctx_t *ctx, char *name, void *smpl_arg, size_t smpl_size)
{
  static int x = 0;
  //printf("create context\n");
  return ++x;
}

int
pfm_write_pmcs(int fd, pfarg_pmc_t *pmcs, int count)
{
  DWORD bytes_returned;
  //uint32_t data[2] = {pmcs, count};
  PVOID data[2] = {pmcs, &count};
  // printf("write pmcs %d %d %llx %d %llx\n", count, pmcs[0].reg_num, pmcs[0].reg_value, pmcs[1].reg_num, pmcs[1].reg_value);
	if (!DeviceIoControl(g_pmc_driver, IOCTL_PMC_WRITE_CONTROL_REGS, &data, sizeof(data), NULL, 0, &bytes_returned, NULL))
	  return PFMLIB_ERR_INVAL;
  return PFMLIB_SUCCESS;
}

int
pfm_write_pmds(int fd, pfarg_pmd_t *pmds, int count)
{
  DWORD bytes_returned;
  //uint32_t data[2] = {pmds, count};
  PVOID data[2] = {pmds, &count};
  //printf("write pmds %d %d %llx %d %llx\n", count, pmds[0].reg_num, pmds[0].reg_value, pmds[1].reg_num, pmds[1].reg_value);

	if (!DeviceIoControl(g_pmc_driver, IOCTL_PMC_WRITE_DATA_REGS, &data, sizeof(data), NULL, 0, &bytes_returned, NULL))
		return PFMLIB_ERR_INVAL;
  return PFMLIB_SUCCESS;
}

int
pfm_read_pmds(int fd, pfarg_pmd_t *pmds, int count)
{
  int i;

	// the counters are 40-bit values for anything less than Pentium 4
	// we mask off the low 8 bits in the high word to exclude possible stray bits
  for (i = 0; i < count; ++i) {
    uint32_t reg;
#ifndef _WIN64
	uint32_t *v;
#endif
    reg = pmds[i].reg_num;
    switch (pmds[i].reg_num)
    {
    case 0:
    case 1:
      reg = pmds[i].reg_num;
      break;
    default:
      goto end;
    }
#ifdef _WIN64
	pmds[i].reg_value = __readpmc(reg);
#else
    v = (uint32_t *)&pmds[i].reg_value;
		__asm
    {
      mov ecx, reg
	    rdpmc
		  mov ebx, v
      mov [ebx], eax
		  and edx, 0x000000FF
		  mov [ebx + 4], edx
    }
#endif
  }
end:
  // supposedly, the fixed function counters are readable by rdpmc in ring 3, but apparantly, Intel messed up
  if (i < count)
  {
    DWORD bytes_returned;
	int xyzzy = count - i;
    //uint32_t data[2] = {&pmds[i], count - i};

	PVOID data[2] = {&pmds[i], &xyzzy};
	if (!DeviceIoControl(g_pmc_driver, IOCTL_PMC_READ_DATA_REGS, &data, sizeof(data), NULL, 0, &bytes_returned, NULL))
      return PFMLIB_ERR_INVAL;
  }
  //printf("pfm read pmds %d %lld\n", count, pmds[0].reg_value);
	return PFMLIB_SUCCESS;
}

int
pfm_load_context(int fd, pfarg_load_t *load)
{
  //printf("pfm load context\n");
	return PFMLIB_SUCCESS;
}

int
pfm_start(int fd, pfarg_start_t *start)
{
  //printf("pfm start\n");
	return PFMLIB_SUCCESS;
}

int
pfm_stop(int fd)
{
  //printf("pfm stop\n");
	return PFMLIB_SUCCESS;
}

int
pfm_restart(int fd)
{
  //printf("pfm restart\n");
	return PFMLIB_SUCCESS;
}

int
pfm_create_evtsets(int fd, pfarg_setdesc_t *setd, int count)
{
  //printf("pfm create_evtsets\n");
	return PFMLIB_SUCCESS;
}

int
pfm_delete_evtsets(int fd, pfarg_setdesc_t *setd, int count)
{
  //printf("pfm delete evtsets\n");
	return PFMLIB_SUCCESS;
}

int
pfm_getinfo_evtsets(int fd, pfarg_setinfo_t *info, int count)
{
  int i;
  pfmlib_regmask_t avail_pmcs, avail_pmds;
  pfm_get_impl_pmcs(&avail_pmcs);
  pfm_get_impl_pmds(&avail_pmds);
    
  memset(info, 0, sizeof(*info) * count);
  for (i = 0; i < count; ++i)
  {
    info[i].set_id = i;
    memcpy(info[i].set_avail_pmcs, &avail_pmcs, sizeof(info[0].set_avail_pmcs));
    memcpy(info[i].set_avail_pmds, &avail_pmds, sizeof(info[0].set_avail_pmds));
    //printf("getinfo_evtsets %llx", info[i].set_avail_pmds[0]);
  }
  return PFMLIB_SUCCESS;
}

int
pfm_unload_context(int fd)
{
  //printf("pfm unload context\n");
	return PFMLIB_SUCCESS;
}

void pfm_init_syscalls()
{
}

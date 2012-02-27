#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#include PEINCLUDE

#include "papi_debug.h" /* SUBDBG */

#include <asm/unistd.h>

/* In case headers aren't new enough to have __NR_perf_event_open */
#ifndef __NR_perf_event_open

#ifdef __powerpc__
#define __NR_perf_event_open	319
#elif defined(__x86_64__)
#define __NR_perf_event_open	298
#elif defined(__i386__)
#define __NR_perf_event_open	336
#elif defined(__arm__)          366+0x900000
#define __NR_perf_event_open    
#endif

#endif

long
sys_perf_event_open( struct perf_event_attr *hw_event, pid_t pid, int cpu,
					   int group_fd, unsigned long flags )
{
	int ret;

   SUBDBG("sys_perf_event_open(%p,%d,%d,%d,%lx\n",hw_event,pid,cpu,group_fd,flags);
   SUBDBG("   type: %d\n",hw_event->type);
   SUBDBG("   size: %d\n",hw_event->size);
   SUBDBG("   config: %"PRIx64" (%"PRIu64")\n",hw_event->config,
	  hw_event->config);
   SUBDBG("   sample_period: %"PRIu64"\n",hw_event->sample_period);
   SUBDBG("   sample_type: %"PRIu64"\n",hw_event->sample_type);
   SUBDBG("   read_format: %"PRIu64"\n",hw_event->read_format);
   SUBDBG("   disabled: %d\n",hw_event->disabled);
   SUBDBG("   inherit: %d\n",hw_event->inherit);
   SUBDBG("   pinned: %d\n",hw_event->pinned);
   SUBDBG("   exclusive: %d\n",hw_event->exclusive);
   SUBDBG("   exclude_user: %d\n",hw_event->exclude_user);
   SUBDBG("   exclude_kernel: %d\n",hw_event->exclude_kernel);
   SUBDBG("   exclude_hv: %d\n",hw_event->exclude_hv);
   SUBDBG("   exclude_idle: %d\n",hw_event->exclude_idle);   
   SUBDBG("   mmap: %d\n",hw_event->mmap);   
   SUBDBG("   comm: %d\n",hw_event->comm);   
   SUBDBG("   freq: %d\n",hw_event->freq);   
   SUBDBG("   inherit_stat: %d\n",hw_event->inherit_stat);   
   SUBDBG("   enable_on_exec: %d\n",hw_event->enable_on_exec);   
   SUBDBG("   task: %d\n",hw_event->task);   
   SUBDBG("   watermark: %d\n",hw_event->watermark);   
   
   
	ret =
		syscall( __NR_perf_event_open, hw_event, pid, cpu, group_fd, flags );
#if defined(__x86_64__) || defined(__i386__)
	if ( ret < 0 && ret > -4096 ) {
		errno = -ret;
		ret = -1;
	}
#endif
	SUBDBG("Returned %d %d %s\n",ret,
	       ret<0?errno:0,
	       ret<0?strerror(errno):" ");
	return ret;
}

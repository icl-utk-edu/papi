#ifndef __MYSIGINFO_H__
#define __MYSIGINFO_H__

typedef union mysigval
  {
    int sival_int;
    void *sival_ptr;
  } mysigval_t;

# define __MYSI_MAX_SIZE     128
# define __MYSI_PAD_SIZE     ((__MYSI_MAX_SIZE / sizeof (int)) - 4)

typedef struct mysiginfo
  {
    int sy_signo;		/* Signal number.  */
    int sy_errno;		/* If non-zero, an errno value associated with
				   this signal, as defined in <errno.h>.  */
    int sy_code;		/* Signal code.  */

    union
      {
	int _pad[__MYSI_PAD_SIZE];
	struct
	  {
	    int sy_pid;
	    int sy_uid;
	    unsigned long sy_pfm_ovfl_counters;
	  } _sigprof;
      } _sifields;
  } mysiginfo_t;

#define sy_pid		_sifields._sigprof.sy_pid
#define sy_pfm_ovfl	_sifields._sigprof.sy_pfm_ovfl_counters

#endif /* __MYSIGINFO_H__ */


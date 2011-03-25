#ifndef __PAPI_COST_UTILS_H__
#define __PAPI_COST_UTILS_H__
extern int num_iters;

extern double	do_stats(long long*, long long*, long long *, double *);
extern void		do_std_dev( long long*, int*, double, double );
extern void		do_dist( long long*, long long, long long, int, int*);

#endif /* __PAPI_COST_UTILS_H__ */

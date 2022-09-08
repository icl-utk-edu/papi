/**
 * @file    htable.h
 * @author  Giuseppe Congiu
 *          gcongiu@icl.utk.edu
 *
 */

#ifndef __HTABLE_H__
#define __HTABLE_H__

#define HTABLE_SUCCESS ( 0)
#define HTABLE_ENOVAL  (-1)
#define HTABLE_EINVAL  (-2)
#define HTABLE_ENOMEM  (-3)

int htable_init(void **handle);
int htable_shutdown(void *handle);
int htable_insert(void *handle, const char *key, void *in);
int htable_delete(void *handle, const char *key);
int htable_find(void *handle, const char *key, void **out);

#endif /* End of __HTABLE_H__ */

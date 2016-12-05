#ifndef APPIO_NOINSTRUMEMENT_INSIDE_PAPI_H
#define APPIO_NOINSTRUMEMENT_INSIDE_PAPI_H

#ifdef linux
#define read(x,y,z) __read(x,y,z)
#define write(x,y,z) __write(x,y,z)
#define select(a,b,c,d,e) __select(a,b,c,d,e)
#define open(...) __open(__VA_ARGS__)
#define close(n) __close(n)
#define lseek(x,y,z) __lseek(x,y,z)

ssize_t __read(int fd, void *buf, size_t count);
ssize_t __write(int fd, const void *buf, size_t count);
int __select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
int __close(int fd);
off_t __lseek(int fd, off_t offset, int whence);
#else
#warning "appio component is tested only on linux"
#endif /* linux */

#endif /* APPIO_NOINSTRUMEMENT_INSIDE_PAPI_H */

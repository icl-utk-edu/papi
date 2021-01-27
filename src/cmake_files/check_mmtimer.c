#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <linux/mmtimer.h>
#ifndef MMTIMER_FULLNAME
#define MMTIMER_FULLNAME "/dev/mmtimer"
#endif

int main() {
  int offset;
  int fd;
  if((fd = open(MMTIMER_FULLNAME, O_RDONLY)) == -1)
    exit(1);
  if ((offset = ioctl(fd, MMTIMER_GETOFFSET, 0)) < 0)
    exit(1);
  close(fd);
  exit(0);
}

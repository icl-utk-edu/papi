#include <sys/types.h>
#include <sys/syscall.h>

int
main(int argc, char *argv[]) {
    pid_t a = syscall(SYS_gettid);
    return 0;
}

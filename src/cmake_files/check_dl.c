#include <dlfcn.h>

int
main(int argc, char *argv[])
{
    void *p = dlopen("", 0);
    char *c = dlerror();
    return 0;
}

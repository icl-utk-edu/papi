#include <stdlib.h>
#include <libcpc.h>

int
main(int argc, char *argv[])
{
    cpc_event_t event;
    return cpc_take_sample(&event);
}

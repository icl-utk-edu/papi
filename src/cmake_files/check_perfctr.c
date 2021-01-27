#include <stdlib.h>
#include "libperfctr.h"

int
main() {
    if ((PERFCTR_ABI_VERSION >> 24) != 5)
        exit (1);
    exit (0);
}

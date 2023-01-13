/**
 * @file   sample_multi_thread_monitoring.cpp
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 */
#include "multi_thread_monitoring.h"

int main(int argc, char *argv[])
{
    test_skip(__FILE__, __LINE__, "sample_multi_thread", PAPI_OK);

    setenv("ROCP_HSA_INTERCEPT", "0", 1);

    multi_thread(argc, argv);

    test_pass(__FILE__);

    return 0;
}

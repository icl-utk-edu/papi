/* flops.c, based on the hl_rates.c ctest
 *
 * This test runs a "classic" matrix multiply
 * and then runs it again with the inner loop swapped.
 * the swapped version should have better MFLIPS/MFLOPS/IPC and we test that.
 */

#include <stdio.h>
#include <stdlib.h>

#include "papi.h"

#include "testcode.h"

int
main( )
{
   // Flips classic
   PAPI_hl_region_begin("matrix_multiply_classic");
   flops_float_matrix_matrix_multiply();
   PAPI_hl_region_end("matrix_multiply_classic");

   // Flips swapped
   PAPI_hl_region_begin("matrix_multiply_swapped");
   flops_float_swapped_matrix_matrix_multiply();
   PAPI_hl_region_end("matrix_multiply_swapped");

   return 0;
}

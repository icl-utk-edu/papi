/* 
* File:    any-null.c
* CVS:     $Id$
* Author:  Philip Mucci
*          mucci@eecs.utk.edu
* Mods:    Kevin London
*          london@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@eecs.utk.edu
* Mods:    Brian Sheely
*          bsheely@eecs.utk.edu
*/

#include "any-null.h"
#include "papi_internal.h"
#include "papi_vector.h"

extern papi_vector_t MY_VECTOR;

papi_vector_t _any_vector = {
	/*Developer's Note: The size data members are set to non-zero values in case 
	   the framework uses them as the size argument to malloc */
	.size = {
			 .context = sizeof ( int ),
			 .control_state = sizeof ( int ),
			 .reg_value = sizeof ( int ),
			 .reg_alloc = sizeof ( int )
			 }
};

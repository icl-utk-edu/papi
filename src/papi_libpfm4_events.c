/*
* File:    papi_libpfm4_events.c
* Author:  Vince Weaver vincent.weaver @ maine.edu
*          based heavily on existing papi_libpfm3_events.c
*/

#include <string.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

#include "papi_libpfm4_events.h"

#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"

/***********************************************************/
/* Exported functions                                      */
/***********************************************************/


/** @class  _papi_libpfm4_error
 *  @brief  convert libpfm error codes to PAPI error codes
 *
 *  @param[in] pfm_error
 *             -- a libpfm4 error code
 *
 *  @returns returns a PAPI error code
 *
 */

int
_papi_libpfm4_error( int pfm_error ) {

  switch ( pfm_error ) {
  case PFM_SUCCESS:      return PAPI_OK;       /* success */
  case PFM_ERR_NOTSUPP:  return PAPI_ENOSUPP;  /* function not supported */
  case PFM_ERR_INVAL:    return PAPI_EINVAL;   /* invalid parameters */
  case PFM_ERR_NOINIT:   return PAPI_ENOINIT;  /* library not initialized */
  case PFM_ERR_NOTFOUND: return PAPI_ENOEVNT;  /* event not found */
  case PFM_ERR_FEATCOMB: return PAPI_ECOMBO;   /* invalid combination of features */
  case PFM_ERR_UMASK:    return PAPI_EATTR;    /* invalid or missing unit mask */
  case PFM_ERR_NOMEM:    return PAPI_ENOMEM;   /* out of memory */
  case PFM_ERR_ATTR:     return PAPI_EATTR;    /* invalid event attribute */
  case PFM_ERR_ATTR_VAL: return PAPI_EATTR;    /* invalid event attribute value */
  case PFM_ERR_ATTR_SET: return PAPI_EATTR;    /* attribute value already set */
  case PFM_ERR_TOOMANY:  return PAPI_ECOUNT;   /* too many parameters */
  case PFM_ERR_TOOSMALL: return PAPI_ECOUNT;   /* parameter is too small */
  default: return PAPI_EINVAL;
  }
}

static int libpfm4_users=0;

/** @class  _papi_libpfm4_shutdown
 *  @brief  Shutdown any initialization done by the libpfm4 code
 *
 *  @retval PAPI_OK       We always return PAPI_OK
 *
 */

int 
_papi_libpfm4_shutdown(void) {


  SUBDBG("Entry\n");

  /* clean out and free the native events structure */
  _papi_hwi_lock( NAMELIB_LOCK );

  libpfm4_users--;

  /* Only free if we're the last user */

  if (!libpfm4_users) {
     pfm_terminate();
  }

  _papi_hwi_unlock( NAMELIB_LOCK );

  return PAPI_OK;
}

/** @class  _papi_libpfm4_init
 *  @brief  Initialize the libpfm4 code
 *
 *  @param[in] my_vector
 *        -- vector of the component doing the initialization
 *
 *  @retval PAPI_OK       We initialized correctly
 *  @retval PAPI_ECMP     There was an error initializing the component
 *
 */

int
_papi_libpfm4_init(papi_vector_t *my_vector) {

   int version;
   pfm_err_t retval = PFM_SUCCESS;

   _papi_hwi_lock( NAMELIB_LOCK );

   if (!libpfm4_users) {
      retval = pfm_initialize();
      if ( retval != PFM_SUCCESS ) libpfm4_users--;
   }
   libpfm4_users++;

   _papi_hwi_unlock( NAMELIB_LOCK );

   if ( retval != PFM_SUCCESS ) {
      PAPIERROR( "pfm_initialize(): %s", pfm_strerror( retval ) );
      return PAPI_ESYS;
   }

   /* get the libpfm4 version */
   SUBDBG( "pfm_get_version()\n");
   if ( (version=pfm_get_version( )) < 0 ) {
      PAPIERROR( "pfm_get_version(): %s", pfm_strerror( retval ) );
      return PAPI_ESYS;
   }

   /* Set the version */
   sprintf( my_vector->cmp_info.support_version, "%d.%d",
	    PFM_MAJ_VERSION( version ), PFM_MIN_VERSION( version ) );

   /* Complain if the compiled-against version doesn't match current version */
   if ( PFM_MAJ_VERSION( version ) != PFM_MAJ_VERSION( LIBPFM_VERSION ) ) {
      PAPIERROR( "Version mismatch of libpfm: compiled %#x vs. installed %#x\n",
				   PFM_MAJ_VERSION( LIBPFM_VERSION ),
				   PFM_MAJ_VERSION( version ) );
      return PAPI_ESYS;
   }

   return PAPI_OK;
}




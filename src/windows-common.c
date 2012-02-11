/*
* File:    windows-common.c
*/

#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <err.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>

#include "papi.h"
#include "papi_internal.h"

#include "linux-memory.h"
#include "linux-common.h"

int
_windows_get_system_info( papi_mdi_t *mdi ) {

	int retval;

        SYSTEM_INFO si;
        HANDLE hModule;
        int len;

        _papi_hwi_system_info.pid = getpid(  );

        hModule = GetModuleHandle( NULL );      // current process
        len =
	  GetModuleFileName( hModule, mdi->exe_info.fullname, PAPI_MAX_STR_LEN );
        if ( len )
	  strcpy( mdi->exe_info.address_info.name, mdi->exe_info.fullname );
	else
	  return ( PAPI_ESYS );

	SUBDBG( "Executable is %s\n", mdi->exe_info.address_info.name );
	SUBDBG( "Full Executable is %s\n", mdi->exe_info.fullname );

	/* Executable regions, may require reading /proc/pid/maps file */

	retval = _linux_update_shlib_info( mdi );
	SUBDBG( "Text: Start %p, End %p, length %d\n",
			mdi->exe_info.address_info.text_start,
			mdi->exe_info.address_info.text_end,
			( int ) ( mdi->exe_info.address_info.text_end -
					  mdi->exe_info.address_info.text_start ) );
	SUBDBG( "Data: Start %p, End %p, length %d\n",
			mdi->exe_info.address_info.data_start,
			mdi->exe_info.address_info.data_end,
			( int ) ( mdi->exe_info.address_info.data_end -
					  mdi->exe_info.address_info.data_start ) );
	SUBDBG( "Bss: Start %p, End %p, length %d\n",
			mdi->exe_info.address_info.bss_start,
			mdi->exe_info.address_info.bss_end,
			( int ) ( mdi->exe_info.address_info.bss_end -
					  mdi->exe_info.address_info.bss_start ) );

	/* PAPI_preload_option information */

	strcpy( mdi->preload_info.lib_preload_env, "LD_PRELOAD" );
	mdi->preload_info.lib_preload_sep = ' ';
	strcpy( mdi->preload_info.lib_dir_env, "LD_LIBRARY_PATH" );
	mdi->preload_info.lib_dir_sep = ':';

	/* Hardware info */

        GetSystemInfo( &si );
        mdi->hw_info.ncpu = mdi->hw_info.totalcpus = si.dwNumberOfProcessors;
        mdi->hw_info.nnodes = 1;

	retval = _windows_get_cpu_info( &mdi->hw_info );
	if ( retval )
		return retval;

	retval = _windows_get_memory_info( &mdi->hw_info, mdi->hw_info.model );
	if ( retval )
		return retval;

	SUBDBG( "Found %d %s(%d) %s(%d) CPUs at %f Mhz, clock %d Mhz.\n",
			mdi->hw_info.totalcpus,
			mdi->hw_info.vendor_string,
			mdi->hw_info.vendor, mdi->hw_info.model_string, mdi->hw_info.model,
			mdi->hw_info.mhz, mdi->hw_info.clock_mhz );

	return PAPI_OK;
}



static void
decode_vendor_string( char *s, int *vendor )
{
	if ( strcasecmp( s, "GenuineIntel" ) == 0 )
		*vendor = PAPI_VENDOR_INTEL;
	else if ( ( strcasecmp( s, "AMD" ) == 0 ) ||
			  ( strcasecmp( s, "AuthenticAMD" ) == 0 ) )
		*vendor = PAPI_VENDOR_AMD;
	else if ( strcasecmp( s, "IBM" ) == 0 )
		*vendor = PAPI_VENDOR_IBM;
	else if ( strcasecmp( s, "Cray" ) == 0 )
		*vendor = PAPI_VENDOR_CRAY;
	else
		*vendor = PAPI_VENDOR_UNKNOWN;
}

extern int __pfm_getcpuinfo_attr( const char *attr, char *ret_buf,
				  size_t maxlen );

int
_windows_get_cpu_info( PAPI_hw_info_t * hw_info )
{
  int retval = PAPI_OK;
  char maxargs[PAPI_HUGE_STR_LEN];
  char *s;
  char model[48];
  int i;
  for ( i = 0; i < 3; ++i )
    __cpuid( &model[i * 16], 0x80000002 + i );
  for ( i = 0; i < 48; ++i )
    model[i] = tolower( model[i] );
  if ( ( s = strstr( model, "mhz" ) ) != NULL ) {
    --s;
    while ( isspace( *s ) || isdigit( *s ) || *s == '.' && s >= model)
      --s;
    ++s;
    hw_info->mhz = ( float ) atof( s ) * 1000;
  } else
    return PAPI_EBUG;

  hw_info->clock_mhz = hw_info->mhz;

  __pfm_getcpuinfo_attr( "vendor_id", hw_info->vendor_string,
			 sizeof ( hw_info->vendor_string ) );

  if ( strlen( hw_info->vendor_string ) )
    decode_vendor_string( hw_info->vendor_string, &hw_info->vendor );
  __cpuid( maxargs, 1 );
  hw_info->revision = *( uint32_t * ) maxargs & 0xf;
  strcpy( hw_info->model_string, model );
  return ( retval );
}





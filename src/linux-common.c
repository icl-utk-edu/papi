/*
* File:    linux-common.c
*/

#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <err.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <syscall.h>
#include <sys/utsname.h>

#include "papi.h"
#include "papi_internal.h"



#include "linux-memory.h"
#include "linux-common.h"

int
_linux_get_system_info( papi_mdi_t *mdi ) {

	int retval;

	char maxargs[PAPI_HUGE_STR_LEN];
	pid_t pid;

	/* Software info */

	/* Path and args */

	pid = getpid(  );
	if ( pid < 0 ) {
		PAPIERROR( "getpid() returned < 0" );
		return PAPI_ESYS;
	}
	mdi->pid = pid;

	sprintf( maxargs, "/proc/%d/exe", ( int ) pid );
	if ( readlink( maxargs, mdi->exe_info.fullname, PAPI_HUGE_STR_LEN ) < 0 ) {
		PAPIERROR( "readlink(%s) returned < 0", maxargs );
		return PAPI_ESYS;
	}

	/* Careful, basename can modify it's argument */

	strcpy( maxargs, mdi->exe_info.fullname );
	strcpy( mdi->exe_info.address_info.name, basename( maxargs ) );

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

	retval = _linux_get_cpu_info( &mdi->hw_info );
	if ( retval )
		return retval;

	retval = _linux_get_memory_info( &mdi->hw_info, mdi->hw_info.model );
	if ( retval )
		return retval;

	SUBDBG( "Found %d %s(%d) %s(%d) CPU's at %f Mhz, clock %d Mhz.\n",
			mdi->hw_info.totalcpus,
			mdi->hw_info.vendor_string,
			mdi->hw_info.vendor, mdi->hw_info.model_string, mdi->hw_info.model,
			mdi->hw_info.mhz, mdi->hw_info.clock_mhz );

	return PAPI_OK;
}

int _linux_detect_nmi_watchdog() {

  int watchdog_detected=0,watchdog_value=0;
  FILE *fff;

  fff=fopen("/proc/sys/kernel/nmi_watchdog","r");
  if (fff!=NULL) {
     if (fscanf(fff,"%d",&watchdog_value)==1) {
        if (watchdog_value>0) watchdog_detected=1;
     }
     fclose(fff);
  }

  return watchdog_detected;
}

int _linux_get_version() {
      
     int major=0,minor=0,sub=0;
     char *ptr;   
     char kernel_name[BUFSIZ];
     struct utsname uname_buffer;

     uname(&uname_buffer); 

     SUBDBG("Native kernel version %s\n",uname_buffer.release);
   
#ifdef ASSUME_KERNEL
     strncpy(kernel_name,ASSUME_KERNEL,BUFSIZ);
     SUBDBG("Assuming kernel version %s\n",kernel_name);
#else
     strncpy(kernel_name,uname_buffer.release,BUFSIZ);
#endif

     ptr=strtok(kernel_name,".");
     if (ptr!=NULL) major=atoi(ptr);
   
     ptr=strtok(NULL,".");
     if (ptr!=NULL) minor=atoi(ptr);
   
     ptr=strtok(NULL,".");
     if (ptr!=NULL) sub=atoi(ptr);
   
     return LINUX_VERSION(major,minor,sub);
}

#define _PATH_SYS_SYSTEM "/sys/devices/system"
#define _PATH_SYS_CPU0	 _PATH_SYS_SYSTEM "/cpu/cpu0"

static char pathbuf[PATH_MAX] = "/";

static FILE *
xfopen( const char *path, const char *mode )
{
	FILE *fd = fopen( path, mode );
	if ( !fd )
		err( EXIT_FAILURE, "error: %s", path );
	return fd;
}

static FILE *
path_vfopen( const char *mode, const char *path, va_list ap )
{
	vsnprintf( pathbuf, sizeof ( pathbuf ), path, ap );
	return xfopen( pathbuf, mode );
}

static int
path_sibling( const char *path, ... )
{
	int c;
	long n;
	int result = 0;
	char s[2];
	FILE *fp;
	va_list ap;
	va_start( ap, path );
	fp = path_vfopen( "r", path, ap );
	va_end( ap );

	while ( ( c = fgetc( fp ) ) != EOF ) {
		if ( isxdigit( c ) ) {
			s[0] = ( char ) c;
			s[1] = '\0';
			for ( n = strtol( s, NULL, 16 ); n > 0; n /= 2 ) {
				if ( n % 2 )
					result++;
			}
		}
	}

	fclose( fp );
	return result;
}

static int
path_exist( const char *path, ... )
{
	va_list ap;
	va_start( ap, path );
	vsnprintf( pathbuf, sizeof ( pathbuf ), path, ap );
	va_end( ap );
	return access( pathbuf, F_OK ) == 0;
}

static char *
search_cpu_info( FILE * f, char *search_str, char *line )
{
	/* This function courtesy of Rudolph Berrendorf! */
	/* See the home page for the German version of PAPI. */
	char *s;

	while ( fgets( line, 256, f ) != NULL ) {
		if ( strstr( line, search_str ) != NULL ) {
			/* ignore all characters in line up to : */
			for ( s = line; *s && ( *s != ':' ); ++s );
			if ( *s )
				return s;
		}
	}
	return NULL;
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
	else if ( strcasecmp( s, "ARM" ) == 0 )
		*vendor = PAPI_VENDOR_ARM;
	else if ( strcasecmp( s, "MIPS" ) == 0 )
		*vendor = PAPI_VENDOR_MIPS;
	else if ( strcasecmp( s, "SiCortex" ) == 0 )
		*vendor = PAPI_VENDOR_MIPS;
	else
		*vendor = PAPI_VENDOR_UNKNOWN;
}


int
_linux_get_cpu_info( PAPI_hw_info_t * hwinfo )
{
  int tmp, retval = PAPI_OK;
	char maxargs[PAPI_HUGE_STR_LEN], *t, *s;
	float mhz = 0.0;
	FILE *f;

	if ( ( f = fopen( "/proc/cpuinfo", "r" ) ) == NULL ) {
		PAPIERROR( "fopen(/proc/cpuinfo) errno %d", errno );
		return PAPI_ESYS;
	}

	/* All of this information maybe overwritten by the substrate */

	/* MHZ */
	rewind( f );
	s = search_cpu_info( f, "clock", maxargs );
	if ( !s ) {
		rewind( f );
		s = search_cpu_info( f, "cpu MHz", maxargs );
	}
	if ( s )
		sscanf( s + 1, "%f", &mhz );
	hwinfo->mhz = mhz;
	hwinfo->clock_mhz = ( int ) mhz;

	/* Vendor Name and Vendor Code */
	rewind( f );
	s = search_cpu_info( f, "vendor_id", maxargs );
	if ( s && ( t = strchr( s + 2, '\n' ) ) ) {
		*t = '\0';
		strcpy( hwinfo->vendor_string, s + 2 );
	} else {
		rewind( f );
		s = search_cpu_info( f, "vendor", maxargs );
		if ( s && ( t = strchr( s + 2, '\n' ) ) ) {
			*t = '\0';
			strcpy( hwinfo->vendor_string, s + 2 );
		} else {
			rewind( f );
			s = search_cpu_info( f, "system type", maxargs );
			if ( s && ( t = strchr( s + 2, '\n' ) ) ) {
				*t = '\0';
				s = strtok( s + 2, " " );
				strcpy( hwinfo->vendor_string, s );
			} else {
				rewind( f );
				s = search_cpu_info( f, "platform", maxargs );
				if ( s && ( t = strchr( s + 2, '\n' ) ) ) {
					*t = '\0';
					s = strtok( s + 2, " " );
					if ( ( strcasecmp( s, "pSeries" ) == 0 ) ||
						 ( strcasecmp( s, "PowerMac" ) == 0 ) ) {
						strcpy( hwinfo->vendor_string, "IBM" );
					}
				 } else {
			            rewind( f );
			            s = search_cpu_info( f, "CPU implementer",
							 maxargs );
			            if ( s ) {
				       strcpy( hwinfo->vendor_string, "ARM" );
				    }
				}

				
			}
		}
	}
	if ( strlen( hwinfo->vendor_string ) )
		decode_vendor_string( hwinfo->vendor_string, &hwinfo->vendor );

	/* Revision */
	rewind( f );
	s = search_cpu_info( f, "stepping", maxargs );
	if ( s ) {
		sscanf( s + 1, "%d", &tmp );
		hwinfo->revision = ( float ) tmp;
		hwinfo->cpuid_stepping = tmp;
	} else {
		rewind( f );
		s = search_cpu_info( f, "revision", maxargs );
		if ( s ) {
			sscanf( s + 1, "%d", &tmp );
			hwinfo->revision = ( float ) tmp;
			hwinfo->cpuid_stepping = tmp;
		}
	}

	/* Model Name */
	rewind( f );
	s = search_cpu_info( f, "model name", maxargs );
	if ( s && ( t = strchr( s + 2, '\n' ) ) ) {
		*t = '\0';
		strcpy( hwinfo->model_string, s + 2 );
	} else {
		rewind( f );
		s = search_cpu_info( f, "family", maxargs );
		if ( s && ( t = strchr( s + 2, '\n' ) ) ) {
			*t = '\0';
			strcpy( hwinfo->model_string, s + 2 );
		} else {
			rewind( f );
			s = search_cpu_info( f, "cpu model", maxargs );
			if ( s && ( t = strchr( s + 2, '\n' ) ) ) {
				*t = '\0';
				strtok( s + 2, " " );
				s = strtok( NULL, " " );
				strcpy( hwinfo->model_string, s );
			} else {
				rewind( f );
				s = search_cpu_info( f, "cpu", maxargs );
				if ( s && ( t = strchr( s + 2, '\n' ) ) ) {
					*t = '\0';
					/* get just the first token */
					s = strtok( s + 2, " " );
					strcpy( hwinfo->model_string, s );
				}
			}
		}
	}

	/* Family */
	rewind( f );
	s = search_cpu_info( f, "family", maxargs );
	if ( s ) {
		sscanf( s + 1, "%d", &tmp );
		hwinfo->cpuid_family = tmp;
	} else {
		rewind( f );
		s = search_cpu_info( f, "cpu family", maxargs );
		if ( s ) {
			sscanf( s + 1, "%d", &tmp );
			hwinfo->cpuid_family = tmp;
		}
	}

	/* CPU Model */
	rewind( f );
	s = search_cpu_info( f, "model", maxargs );
	if ( s ) {
		sscanf( s + 1, "%d", &tmp );
		hwinfo->model = tmp;
		hwinfo->cpuid_model = tmp;
	}

	fclose( f );
	/* The following new members are set using the same methodology used in lscpu. */

	/* Total number of CPUs */
	/* The following line assumes totalcpus was initialized to zero! */
	while ( path_exist( _PATH_SYS_SYSTEM "/cpu/cpu%d", hwinfo->totalcpus ) )
		hwinfo->totalcpus++;

	/* Number of threads per core */
	if ( path_exist( _PATH_SYS_CPU0 "/topology/thread_siblings" ) )
		hwinfo->threads =
			path_sibling( _PATH_SYS_CPU0 "/topology/thread_siblings" );

	/* Number of cores per socket */
	if ( path_exist( _PATH_SYS_CPU0 "/topology/core_siblings" ) &&
		 hwinfo->threads > 0 )
		hwinfo->cores =
			path_sibling( _PATH_SYS_CPU0 "/topology/core_siblings" ) /
			hwinfo->threads;

	/* Number of sockets */
	if ( hwinfo->threads > 0 && hwinfo->cores > 0 )
		hwinfo->sockets = hwinfo->ncpu / hwinfo->cores / hwinfo->threads;

	/* Number of NUMA nodes */
	/* The following line assumes nnodes was initialized to zero! */
	while ( path_exist( _PATH_SYS_SYSTEM "/node/node%d", hwinfo->nnodes ) )
		hwinfo->nnodes++;

	/* Number of CPUs per node */
	hwinfo->ncpu =
		hwinfo->nnodes >
		1 ? hwinfo->totalcpus / hwinfo->nnodes : hwinfo->totalcpus;
#if 0
	int *nodecpu;
	/* cpumap data is not currently part of the _papi_hw_info struct */
        nodecpu = malloc( (unsigned int) hwinfo->nnodes * sizeof(int) );
	if ( nodecpu ) {
	   int i;
	   for ( i = 0; i < hwinfo->nnodes; ++i ) {
	       nodecpu[i] = path_sibling( 
                             _PATH_SYS_SYSTEM "/node/node%d/cpumap", i );
	   }
	} else {
		PAPIERROR( "malloc failed for variable not currently used" );
	}
#endif

	return retval;
}

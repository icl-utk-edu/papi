#include <host_counter.h>
#include <stdio.h>
#include <sys/types.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <fcntl.h>
#include <errno.h>
#include <assert.h>
#include <dirent.h>
#include <config.h>

#ifdef USE_INFINIBAND

#define __BUILD_VERSION_TAG__ 1.2
#include <infiniband/common.h>
#include <infiniband/umad.h>
#include <infiniband/mad.h>
#include <stdarg.h>

#endif

static int    is_initialized = 0;
static int    is_finalized   = 0;
static int    num_counters   = 0;
static int    is_ready       = 0;
static char*  buffer         = NULL;
static size_t pagesz         = 0;

static FILE*  proc_fd_snmp = NULL;
static FILE*  proc_fd_dev  = NULL;

/**
 * describes a single counter with its properties as
 * it is being presented to VT
 */
/*typedef struct counter_info_struct
{
    char*    name;
    char*    description;
    char*    unit;
    uint64_t value;
    struct counter_info_struct* next;
} counter_info;
*/
/**
 * counters are kept in a list
 */
static counter_info* root_counter = NULL;

/**
 * some counters that are always present
 */
static counter_info* tcp_sent     = NULL;
static counter_info* tcp_recv     = NULL;
static counter_info* tcp_retr     = NULL;

/**
 * describes the infos collected from a mounted
 * Lustre fs
 */
typedef struct lustre_fs_struct
{
    FILE*         proc_fd;
    FILE*         proc_fd_readahead;
    counter_info* write_cntr;
    counter_info* read_cntr;
    counter_info* readahead_cntr;
    struct lustre_fs_struct* next;
} lustre_fs;

/**
 * mount Lustre fs are kept in a list
 */
static lustre_fs* root_lustre_fs = NULL;

/**
 * describes one network interface
 */
typedef struct network_if_struct
{
    char*         name;
    counter_info* send_cntr;
    counter_info* recv_cntr;
    struct network_if_struct* next;
} network_if;

/**
 * network interfaces are kept in a list as well
 */
static network_if* root_network_if = NULL;

#ifdef USE_INFINIBAND
/**
 * infos collected of a single IB port
 */
typedef struct ib_port_struct
{
    char*         name;
    counter_info* send_cntr;
    counter_info* recv_cntr;
    int           port_rate;
    int           port_number;
    int           is_initialized;
    uint64_t      sum_send_val;
    uint64_t      sum_recv_val;
    uint32_t      last_send_val;
    uint32_t      last_recv_val;

    struct ib_port_struct* next;
} ib_port;

/**
 * IB ports found are kept in a list
 */
static ib_port* root_ib_port   = NULL;
/**
 * as of now there can only one active IB port
 * never had the time to test two used at the same
 * time
 */
static ib_port* active_ib_port = NULL;

static void init_ib_counter();
static int read_ib_counter();
static int init_ib_port( ib_port* portdata );
static void addIBPort( const char*  ca_name,
                       umad_port_t* port );

static ib_portid_t portid = { 0 };
static int         ib_timeout = 0;
static int         ibportnum  = 0;
#endif

counter_info* subscriptions[ MAX_SUBSCRIBED_COUNTER ];

/**
 * open a file and return the FILE handle
 * @param
 */
static FILE* proc_fopen( const char* procname )
{
    FILE* fd = fopen( procname, "r" );

    if ( fd == NULL )
    {
        /* perror("unable to open proc file"); */
        return NULL;
    }

    if ( !buffer )
    {
        pagesz = getpagesize();
        buffer = malloc( pagesz );
        memset( buffer, 0, pagesz );
    }

    setvbuf( fd, buffer, _IOFBF, pagesz );
    return fd;
}

/**
 * read one PAGE_SIZE byte from a FILE handle
 * and reset the file pointer
 */
static void readPage( FILE* fd )
{
    int count_read;
    count_read = fread( buffer, pagesz, 1, fd );
    if ( fseek( fd, 0L, SEEK_SET ) != 0 )
    {
        fprintf( stderr, "can not seek back in proc\n" );
        exit( 1 );
    }
}

/**
 * add a counter to the list of available counters
 * @param name the short name of the counter
 * @param desc a longer description
 * @param unit the unit for this counter
 */
static counter_info* addCounter( const char* name,
                                 const char* desc,
                                 const char* unit )
{
    counter_info* cntr, * last;

    cntr = ( counter_info* )malloc( sizeof( counter_info ));
    if ( cntr == NULL )
    {
        fprintf( stderr, "can not allocate memory for new counter\n" );
        exit( 1 );
    }
    cntr->name        = strdup( name );
    cntr->description = strdup( desc );
    cntr->unit        = strdup( unit );
    cntr->value       = 0;
    cntr->next        = NULL;

    if ( root_counter == NULL )
    {
        root_counter = cntr;
    }
    else
    {
        last = root_counter;
        while ( last->next != NULL )
        {
            last = last->next;
        }
        last->next = cntr;
    }

    return cntr;
}

/**
 * adds a network interface to the list of available
 * interfaces and the counters to the list of available
 * counters
 */
static void addNetworkIf( const char* name )
{
    network_if* nwif, * last;
    char        counter_name[ 512 ];

    nwif = ( network_if* )malloc( sizeof( network_if ));
    if ( nwif == NULL )
    {
        fprintf(
            stderr,
            "can not allocate memory for new network interface description\n" );
        exit( 1 );
    }

    nwif->name = strdup( name );
    sprintf( counter_name, "%s_recv", name );
    nwif->recv_cntr =
        addCounter( counter_name, "bytes received on this interface",
                    "bytes" );
    sprintf( counter_name, "%s_send", name );
    nwif->send_cntr =
        addCounter( counter_name, "bytes written on this interface",
                    "bytes" );
    nwif->next = NULL;

    num_counters += 2;

    if ( root_network_if == NULL )
    {
        root_network_if = nwif;
    }
    else
    {
        last = root_network_if;
        while ( last->next != NULL )
        {
            last = last->next;
        }
        last->next = nwif;
    }
}

/**
 * looks after available IP interfaces/cards
 */
static void init_tcp_counter()
{
    char* ptr;
    char  name[ 100 ];
    int   idx;

    /* init the static stuff from /proc/net/snmp */
    proc_fd_snmp = proc_fopen( "/proc/net/snmp" );
    if ( proc_fd_snmp == NULL )
    {
        return;
        /* fprintf( stderr, "can not open /proc/net/snmp\n");
           exit(1);
         */
    }

    tcp_sent = addCounter( "tcp_segments_sent", "# of TCP segments sent",
                           "segments" );
    tcp_recv =
        addCounter( "tcp_segments_received", "# of TCP segments received",
                    "segments" );
    tcp_retr =
        addCounter( "tcp_segments_retransmitted",
                    "# of TCP segments retransmitted",
                    "segments" );

    num_counters += 3;

    /* now the individual interfaces */
    proc_fd_dev = proc_fopen( "/proc/net/dev" );
    if ( proc_fd_dev == NULL )
    {
        return;
        /* fprintf( stderr, "can not open /proc/net/dev\n");
           exit(1);
         */
    }

    readPage( proc_fd_dev );
    ptr = buffer;
    while ( *ptr != 0 )
    {
        while ( *ptr != 0 && *ptr != ':' )
        {
            ptr++;
        }
        if ( *ptr == 0 )
        {
            break;
        }

        // move backwards until space or '\n'
        while ( *ptr != ' ' && *ptr != '\n' )
        {
            ptr--;
        }
        ptr++;

        memset( name, 0, sizeof( name ));
        idx = 0;
        while ( *ptr != ':' )
        {
            name[ idx++ ] = *ptr++;
        }
        ptr++;

        // printf("new interface: '%s'\n", name);

        addNetworkIf( name );
    }
}

/**
 * reads all cards and updates the associated
 * counters
 */
static void read_tcp_counter()
{
    int64_t     in, out, retr;
    char*       ptr;
    int         num;
    int         idx;
    char        name[ 100 ];
    network_if* current_if = root_network_if;

    if ( proc_fd_snmp != NULL )
    {
        readPage( proc_fd_snmp );

        ptr  = strstr( buffer, "Tcp:" );
        ptr += 4;
        ptr  = strstr( ptr, "Tcp:" );
        ptr += 4;
        num  = 0;
        while ( num < 10 )
        {
            if ( *ptr == ' ' )
            {
                num++;
            }
            ptr++;
        }

        in = strtoll( ptr, NULL, 10 );
        while ( *ptr != ' ' )
        {
            ptr++;
        }
        ptr++;
        out = strtoll( ptr, NULL, 10 );
        while ( *ptr != ' ' )
        {
            ptr++;
        }
        ptr++;
        retr = strtoll( ptr, NULL, 10 );

        tcp_sent->value = out;
        tcp_recv->value = in;
        tcp_retr->value = retr;
    }

    if ( proc_fd_dev != NULL )
    {
        /* now parse /proc/net/dev */
        readPage( proc_fd_dev );
        ptr = buffer;
        // jump over first two \n
        while ( *ptr != 0 && *ptr != '\n' )
        {
            ptr++;
        }
        if ( *ptr == 0 )
        {
            return;
        }
        ptr++;
        while ( *ptr != 0 && *ptr != '\n' )
        {
            ptr++;
        }
        if ( *ptr == 0 )
        {
            return;
        }
        ptr++;

        while ( *ptr != 0 )
        {
            if ( current_if == NULL )
            {
                break;
            }
            // move to next non space char
            while ( *ptr == ' ' )
            {
                ptr++;
            }
            if ( *ptr == 0 )
            {
                return;
            }

            // copy name until ':'
            idx = 0;
            while ( *ptr != ':' )
            {
                name[ idx++ ] = *ptr++;
            }
            if ( *ptr == 0 )
            {
                return;
            }
            name[ idx ] = 0;

            // compare and make sure network interface are still
            // showing up in the same order. adding or deleting
            // some or changing the order during the run is not
            // support yet (overhead)
            if ( current_if == NULL )
            {
                fprintf( stderr, "error: current interface is NULL\n" );
                exit( 1 );
            }
            if ( strcmp( name, current_if->name ) != 0 )
            {
                fprintf(
                    stderr,
                    "wrong interface, order changed(?): got %s, wanted %s\n",
                    name, current_if->name );
                exit( 1 );
            }

            // move forward to next number
            while ( *ptr<'0' || * ptr>'9' )
            {
                ptr++;
            }
            if ( *ptr == 0 )
            {
                return;
            }

            in = strtoll( ptr, NULL, 10 );

            // move eight numbers forward
            for ( num = 0; num < 8; num++ )
            {
                // move to next space
                while ( *ptr != ' ' )
                {
                    ptr++;
                }
                if ( *ptr == 0 )
                {
                    return;
                }
                // move forward to next number
                while ( *ptr<'0' || * ptr>'9' )
                {
                    ptr++;
                }
                if ( *ptr == 0 )
                {
                    return;
                }
            }

            out = strtoll( ptr, NULL, 10 );

            // move to next newline
            while ( *ptr != '\n' )
            {
                ptr++;
            }
            ptr++;

            current_if->recv_cntr->value = in;
            current_if->send_cntr->value = out;

            current_if = current_if->next;
        }
    }
}

/**
 * adds a Lustre fs to the fs list and creates the counters
 * for it
 * @param name fs name
 * @param procpath_general path to the 'stats' file in /proc/fs/lustre/... for this fs
 * @param procpath_readahead path to the 'readahead' file in /proc/fs/lustre/... for this fs
 */
static void addLustreFS( const char* name,
                         const char* procpath_general,
                         const char* procpath_readahead )
{
    lustre_fs* fs, * last;
    char       counter_name[ 512 ];

    fs = ( lustre_fs* )malloc( sizeof( lustre_fs ));
    if ( fs == NULL )
    {
        fprintf( stderr,
                 "can not allocate memory for new Lustre FS description\n" );
        exit( 1 );
    }

    fs->proc_fd = proc_fopen( procpath_general );
    if ( fs->proc_fd == NULL )
    {
        fprintf( stderr, "can not open '%s'\n", procpath_general );
        exit( 1 );
    }

    fs->proc_fd_readahead = proc_fopen( procpath_readahead );
    if ( fs->proc_fd_readahead == NULL )
    {
        fprintf( stderr, "can not open '%s'\n", procpath_readahead );
        exit( 1 );
    }

    sprintf( counter_name, "%s_llread", name );
    fs->read_cntr =
        addCounter( counter_name, "bytes read on this lustre client",
                    "bytes" );
    sprintf( counter_name, "%s_llwrite", name );
    fs->write_cntr =
        addCounter( counter_name, "bytes written on this lustre client",
                    "bytes" );
    sprintf( counter_name, "%s_wrong_readahead", name );
    fs->readahead_cntr = addCounter(
        counter_name, "bytes read but discarded due to readahead", "bytes" );
    fs->next = NULL;

    num_counters += 3;

    if ( root_lustre_fs == NULL )
    {
        root_lustre_fs = fs;
    }
    else
    {
        last = root_lustre_fs;
        while ( last->next != NULL )
        {
            last = last->next;
        }
        last->next = fs;
    }
}

/**
 * goes through proc and tries to discover all mounted Lustre fs
 */
static void init_lustre_counter()
{
    const char*    proc_base_path = "/proc/fs/lustre/llite";
    char           path[ PATH_MAX ];
    char           path_readahead[ PATH_MAX ];
    char*          ptr;
    char           fs_name[ 100 ];
    int            idx = 0;
    int            tmp_fd;
    DIR*           proc_fd;
    struct dirent* entry;

    proc_fd = opendir( proc_base_path );
    if ( proc_fd == NULL )
    {
        // we are not able to read this directory ...
        return;
    }

    entry = readdir( proc_fd );
    while ( entry != NULL )
    {
        memset( path, 0, PATH_MAX );
        snprintf( path, PATH_MAX-1, "%s/%s/stats", proc_base_path,
                  entry->d_name );
        //fprintf( stderr, "checking for file %s\n", path);
        if (( tmp_fd = open( path, O_RDONLY )) != -1 )
        {
            close( tmp_fd );
            // erase \r and \n at the end of path
            idx = strlen( path );
            idx--;
            while ( path[ idx ] == '\r' || path[ idx ] == '\n' )
            {
                path[ idx-- ] = 0;
            }
            //  /proc/fs/lustre/llite/ has a length of 22 byte
            memset( fs_name, 0, 100 );
            idx = 0;
            ptr = &path[ 22 ];
            while ( *ptr != '-' && idx < 100 )
            {
                fs_name[ idx ] = *ptr;
                ptr++;
                idx++;
            }
            /* printf("found Lustre FS: %s\n", fs_name); */
            strncpy( path_readahead, path, PATH_MAX );
            ptr = strrchr( path_readahead, '/' );
            if ( ptr == NULL )
            {
                fprintf( stderr, "no slash in %s ?\n", path_readahead );
                fflush( stderr );
                exit( 1 );
            }
            ptr++;
            strcpy( ptr, "read_ahead_stats" );
            addLustreFS( fs_name, path, path_readahead );

            memset( path, 0, PATH_MAX );
        }
        entry = readdir( proc_fd );
    }

    closedir( proc_fd );
}

/**
 * updates all Lustre related counters
 */
static void read_lustre_counter()
{
    char*      ptr;

    lustre_fs* fs = root_lustre_fs;

    while ( fs != NULL )
    {
        readPage( fs->proc_fd );

        ptr = strstr( buffer, "write_bytes" );
        if ( ptr == NULL )
        {
            fs->write_cntr->value = 0;
        }
        else
        {
            /* goto eol */
            while ( *ptr != '\n' )
            {
                ptr++;
            }
            *ptr = 0;
            while ( *ptr != ' ' )
            {
                ptr--;
            }
            ptr++;
            fs->write_cntr->value = strtoll( ptr, NULL, 10 );
        }

        ptr = strstr( buffer, "read_bytes" );
        if ( ptr == NULL )
        {
            fs->read_cntr->value = 0;
        }
        else
        {
            /* goto eol */
            while ( *ptr != '\n' )
            {
                ptr++;
            }
            *ptr = 0;
            while ( *ptr != ' ' )
            {
                ptr--;
            }
            ptr++;
            fs->read_cntr->value = strtoll( ptr, NULL, 10 );
        }

        readPage( fs->proc_fd_readahead );
        ptr = strstr( buffer, "read but discarded" );
        if ( ptr == NULL )
        {
            fs->write_cntr->value = 0;
        }
        else
        {
            /* goto next number */
            while ( *ptr<'0' || * ptr>'9' )
            {
                ptr++;
            }
            fs->readahead_cntr->value = strtoll( ptr, NULL, 10 );
        }

        fs = fs->next;
    }
}

/**
 * initializes this library
 */
void host_initialize()
{
    int loop;

    if ( is_initialized )
    {
        return;
    }
    is_initialized = 1;

#ifdef USE_INFINIBAND
    init_ib_counter();
#endif
    init_tcp_counter();
    init_lustre_counter();

    for ( loop = 0; loop < MAX_SUBSCRIBED_COUNTER; loop++ )
    {
        subscriptions[ loop ] = NULL;
    }

    is_ready = 1;
}

/**
 * finalizes the library
 */
void host_finalize()
{
    lustre_fs*    fs, * next_fs;
    counter_info* cntr, * next;
    network_if*   nwif, * next_nwif;

    if ( is_finalized )
    {
        return;
    }

    if ( proc_fd_snmp != NULL )
    {
        fclose( proc_fd_snmp );
    }
    if ( proc_fd_dev != NULL )
    {
        fclose( proc_fd_dev );
    }
    proc_fd_snmp = NULL;
    proc_fd_dev  = NULL;

    if ( buffer != NULL )
    {
        free( buffer );
    }
    buffer = NULL;

    cntr = root_counter;

    while ( cntr != NULL )
    {
        next = cntr->next;
        free( cntr->name );
        free( cntr->description );
        free( cntr->unit );
        free( cntr );
        cntr = next;
    }
    root_counter = NULL;

    fs = root_lustre_fs;
    while ( fs != NULL )
    {
        next_fs = fs->next;
        free( fs );
        fs = next_fs;
    }
    root_lustre_fs = NULL;

    nwif = root_network_if;
    while ( nwif != NULL )
    {
        next_nwif = nwif->next;
        free( nwif->name );
        free( nwif );
        nwif = next_nwif;
    }
    root_network_if = NULL;

    is_finalized = 1;
}

/**
 * read all values for all counters
 */
void host_read_values( long long* data )
{
    int loop;
    read_tcp_counter();
    read_lustre_counter();
#ifdef USE_INFINIBAND
    read_ib_counter();
#endif

    for ( loop = 0; loop < MAX_SUBSCRIBED_COUNTER; loop++ )
    {
        if ( subscriptions[ loop ] == NULL )
        {
            break;
        }
        data[ loop ] = subscriptions[ loop ]->value;
    }
}

/**
 * delete a list of strings
 */
void host_deleteStringList( string_list* to_delete )
{
    int loop;

    if ( to_delete->data != NULL )
    {
        for ( loop = 0; loop < to_delete->count; loop++ )
        {
            free( to_delete->data[ loop ] );
        }
        free( to_delete->data );
    }
    free( to_delete );
}

/**
 * return a newly allocated list of strings containing all
 * counter names
 */
string_list* host_listCounter()
{
    string_list*  list;
    counter_info* cntr = root_counter;

    list = malloc( sizeof( string_list ));
    if ( list == NULL )
    {
        fprintf( stderr, "unable to allocate memory for new string_list" );
        exit( 1 );
    }
    list->count = 0;
    list->data  = ( char** )malloc( num_counters*sizeof( char* ));
    if ( list->data == NULL )
    {
        fprintf(
            stderr,
            "unable to allocate memory for %d pointers in a new string_list\n",
            num_counters );
        exit( 1 );
    }

    while ( cntr != NULL )
    {
        list->data[ list->count++ ] = strdup( cntr->name );
        cntr                        = cntr->next;
    }

    return list;
}

/**
 * find the pointer for a counter_info structure based on the
 * counter name
 */
counter_info* counterFromName( const char* cntr )
{
    int           loop = 0;
    char          tmp[ 512 ];
    counter_info* local_cntr = root_counter;
    while ( local_cntr != NULL )
    {
        if ( strcmp( cntr, local_cntr->name ) == 0 )
        {
            return local_cntr;
        }
        local_cntr = local_cntr->next;
        loop++;
    }
    gethostname( tmp, 512 );
    fprintf( stderr, "can not find host counter: %s on %s\n", cntr, tmp );
    fprintf( stderr, "we only have: " );
    local_cntr = root_counter;
    while ( local_cntr != NULL )
    {
        fprintf( stderr, "'%s' ", local_cntr->name );
        local_cntr = local_cntr->next;
        loop++;
    }
    fprintf( stderr, "\n" );
    exit( 1 );
    /* never reached */
    return 0;
}

/**
 * allow external code to subscribe to a counter based on the counter
 * name
 */
uint64_t host_subscribe( const char* cntr )
{
    int           loop;
#ifdef USE_INFINIBAND
    int           len;
    char          tmp_name[ 512 ];
    ib_port*      aktp;
#endif
    counter_info* counter = counterFromName( cntr );

    for ( loop = 0; loop < MAX_SUBSCRIBED_COUNTER; loop++ )
    {
        if ( subscriptions[ loop ] == NULL )
        {
            subscriptions[ loop ] = counter;
	    counter->idx = loop;
            //fprintf( stderr, "subscription %d is %s\n", loop, subscriptions[ loop ]->name);
#ifdef USE_INFINIBAND
            // we have an IB counter if the name ends with _send or _recv and
            // the prefix before that is in the ib_port list
            if (( len = strlen( cntr )) > 5 )
            {
                if ( strcmp( &cntr[ len-5 ], "_recv" ) == 0 ||
                     strcmp( &cntr[ len-5 ], "_send" ) == 0 )
                {
                    // look through all IB_counters
                    strncpy( tmp_name, cntr, len-5 );
                    tmp_name[ len-5 ] = 0;
                    aktp              = root_ib_port;
                    // printf("looking for IB port '%s'\n", tmp_name);
                    while ( aktp != NULL )
                    {
                        if ( strcmp( aktp->name, tmp_name ) == 0 )
                        {
                            if( !aktp->is_initialized )
                            {
                                init_ib_port( aktp );
                                active_ib_port = aktp;
                            }
                            return loop+1;
                        }
                        // name does not match, if this counter is
                        // initialized, we can't have two active IB ports
                        if ( aktp->is_initialized )
                        {
                            fprintf(
                                stderr,
                                "unable to activate IB port monitoring for more than one port\n" );
                            exit( 1 );
                        }
                        aktp = aktp->next;
                    }
                }
            }
#endif
            return loop+1;
        }
    }
    fprintf( stderr, "please subscribe only once to each counter\n" );
    exit( 1 );
    /* never reached */
    return 0;
}

/**
 * return the description of a counter
 */
const char* host_description( const char* cntr )
{
    counter_info* counter = counterFromName( cntr );
    return counter->description;
}

/**
 * return the unit of a counter
 */
const char* host_unit( const char* cntr )
{
    counter_info* counter = counterFromName( cntr );
    return counter->unit;
}

#ifdef USE_INFINIBAND

#define IBERROR( fmt, args... )   iberror( __FUNCTION__, fmt, ## args )

/**
 * print error and die
 */
static void iberror( const char* fn,
                     char*       msg,
                     ... )
{
    char    buf[ 512 ];
    va_list va;
    int     n;

    va_start( va, msg );
    n = vsprintf( buf, msg, va );
    va_end( va );
    buf[ n ] = 0;

    printf( "iberror: failed: %s\n", buf );
    exit( -1 );
}

/**
 * use libumad to discover IB ports
 */
static void init_ib_counter()
{
    char      names[ 20 ][ UMAD_CA_NAME_LEN ];
    int       n, i;
    char*     ca_name;
    umad_ca_t ca;
    int       r;
    int       portnum;

    if ( umad_init() < 0 )
    {
        IBERROR( "can't init UMAD library" );
    }

    if (( n = umad_get_cas_names(( void* )names, UMAD_CA_NAME_LEN )) < 0 )
    {
        IBERROR( "can't list IB device names" );
    }

    for ( i = 0; i < n; i++ )
    {
        ca_name = names[ i ];

        if (( r = umad_get_ca( ca_name, &ca )) < 0 )
        {
            IBERROR( "can't read ca from IB device" );
        }

        if ( !ca.node_type )
        {
            continue;
        }

        // port numbers are '1' based in OFED
        for ( portnum = 1; portnum <= ca.numports; portnum++ )
        {
            addIBPort( ca.ca_name, ca.ports[ portnum ] );
        }
    }
}

/**
 * add one IB port to the list of available ports and add the
 * counters related to this port to the global counter list
 */
static void addIBPort( const char*  ca_name,
                       umad_port_t* port )
{
    ib_port* nwif, * last;
    char counter_name[ 512 ];

    nwif = ( ib_port* )malloc( sizeof( ib_port ));
    if ( nwif == NULL )
    {
        fprintf( stderr, "can not allocate memory for IB port description\n" );
        exit( 1 );
    }

    sprintf( counter_name, "%s_%d", ca_name, port->portnum );
    nwif->name = strdup( counter_name );
    sprintf( counter_name, "%s_%d_recv", ca_name, port->portnum );
    nwif->recv_cntr =
        addCounter( counter_name, "bytes received on this IB port",
                    "bytes" );
    sprintf( counter_name, "%s_%d_send", ca_name, port->portnum );
    nwif->send_cntr =
        addCounter( counter_name, "bytes written to this IB port",
                    "bytes" );
    nwif->port_rate      = port->rate;
    nwif->is_initialized = 0;
    nwif->port_number    = port->portnum;
    nwif->next           = NULL;

    num_counters += 2;

    if ( root_ib_port == NULL )
    {
        root_ib_port = nwif;
    }
    else
    {
        last = root_ib_port;
        while ( last->next != NULL )
        {
            last = last->next;
        }
        last->next = nwif;
    }
}

/**
 * initialize one IB port so that we are able to read values from it
 */
static int init_ib_port( ib_port* portdata  )
{
    int            mgmt_classes[ 4 ] =
    { IB_SMI_CLASS, IB_SMI_DIRECT_CLASS, IB_SA_CLASS, IB_PERFORMANCE_CLASS };
    char*          ca = 0;
    static uint8_t pc[ 1024 ];
    int      mask        = 0xFFFF;

    madrpc_init( ca, portdata->port_number, mgmt_classes, 4 );
    // printf( "init IB port %d\n", portdata->port_number );

    if ( ib_resolve_self( &portid, &ibportnum, 0 ) < 0 )
    {
        IBERROR( "can't resolve self port" );
    }

    /* PerfMgt ClassPortInfo is a required attribute */
    /* might be redundant, could be left out for fast implementation */
    if ( !perf_classportinfo_query( pc, &portid, ibportnum, ib_timeout ))
    {
        IBERROR( "classportinfo query" );
    }

    if ( !port_performance_reset( pc, &portid, ibportnum, mask, ib_timeout ))
    {
        IBERROR( "perfquery" );
    }

    // read the initial values
    mad_decode_field( pc, IB_PC_XMT_BYTES_F, &portdata->last_send_val );
    portdata->sum_send_val = 0;
    mad_decode_field( pc, IB_PC_RCV_BYTES_F, &portdata->last_recv_val );
    portdata->sum_recv_val = 0;

    portdata->is_initialized = 1;

    return 0;
}

/**
 * read and reset IB counters (reset on demand)
 */
static int read_ib_counter()
{
    uint32_t send_val;
    uint32_t recv_val;
    uint8_t  pc[ 1024 ];
    // 32 bit counter
    uint32_t max_val     = 4294967295;
    // if it it bigger than this -> reset
    uint32_t reset_limit = max_val*0.7;
    int      mask        = 0xFFFF;

    /* printf("%p\n", active_ib_port); */

    if ( active_ib_port == NULL )
    {
        return 0;
    }

    // reading cost ~70 mirco secs
    if ( !port_performance_query( pc, &portid, ibportnum, ib_timeout ))
    {
        IBERROR( "perfquery" );
    }
    mad_decode_field( pc, IB_PC_XMT_BYTES_F, &send_val );
    mad_decode_field( pc, IB_PC_RCV_BYTES_F, &recv_val );

    // multiply the numbers read by 4 as the IB port counters are not
    // counting bytes. they always count 32dwords. see man page of
    // perfquery for details
    // internally a uint64_t ia used to sum up the values
    active_ib_port->sum_send_val +=
        ( send_val-active_ib_port->last_send_val )*4;
    active_ib_port->sum_recv_val +=
        ( recv_val-active_ib_port->last_recv_val )*4;

    /*printf( "s: %10llu    r: %10llu\n",
            (long long unsigned) active_ib_port->sum_send_val,
            (long long unsigned) active_ib_port->sum_recv_val);*/

    active_ib_port->send_cntr->value = active_ib_port->sum_send_val;
    active_ib_port->recv_cntr->value = active_ib_port->sum_recv_val;

    if ( send_val > reset_limit || recv_val > reset_limit )
    {
        // reset cost ~70 mirco secs
        if ( !port_performance_reset( pc, &portid, ibportnum, mask, ib_timeout ))
        {
            IBERROR( "perf reset" );
        }
        mad_decode_field( pc, IB_PC_XMT_BYTES_F, &active_ib_port->last_send_val );
        mad_decode_field( pc, IB_PC_RCV_BYTES_F, &active_ib_port->last_recv_val );
    }
    else
    {
        active_ib_port->last_send_val = send_val;
        active_ib_port->last_recv_val = recv_val;
    }
    return 0;
}
#endif

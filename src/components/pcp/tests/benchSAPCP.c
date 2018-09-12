//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// This bench interacts with PCP directly from a main(), to time reading an
// individual PCP event, given on the command line. e.g.
// ./benchSAPCP "perfevent.hwcounters.instructions.value"
//
// We do the work of extracting the first value (PCP allows an array of values
// to be returned).
//
// We will printf() the initialization time and measurement time on the same
// line in CSV format.  If no arguments are given, we will printf() a header
// CSV line. Otherwise there must be exactly one argument. Errors are printed
// to 'stderr'. This scheme allows a shell loop to produce a csv file with a
// header with many samples, to be processed separately (by spreadsheet or
// another program).
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>
#include <dlfcn.h>      // Dynamic lib routines; especially dlsym to get func ptrs.

// Event Name filters, used in init_component for the routine pmTraversePMNS(). Differs by system.
//#define AGENT_NAME "xfs"          /* Saturn PCP. */
//#define AGENT_NAME "mem"          /* Saturn PCP. */
  #define AGENT_NAME "perfevent"    /* Power9 PCP. */
//#define AGENT_NAME ""             /* Get it all! */

/* To remove redefined warnings */
//#undef PACKAGE_BUGREPORT
//#undef PACKAGE_TARNAME
//#undef PACKAGE_NAME
//#undef PACKAGE_STRING
//#undef PACKAGE_VERSION

#include <pcp/pmapi.h> // See https://pcp.io/man/man3/pmapi.3.html for routines.
#include <pcp/impl.h>  // also a PCP file.
int retZero(unsigned long long);                            // returns a zero alone. 

#define MYPCPLIB "libpcp.so"  // Name of my PCP library. 

#define   PM_OPTFLAG_EXIT     (1<<5)
#define   PM_CONTEXT_UNDEF    -1
#define   PM_CONTEXT_HOST     1
#define   PM_CONTEXT_ARCHIVE  2
#define   PM_CONTEXT_LOCAL    3

//-----------------------------------------------------------------------------
// Union to convert pointers and avoid warnings. Plug in one, pull out the other.
//-----------------------------------------------------------------------------
typedef union  
{
   void                 *vPtr;
   int                  *iPtr;
   unsigned int         *uiPtr;
   long                 *lPtr;
   long long            *llPtr;
   unsigned long long   *ullPtr;
   float                *fPtr;
   double               *dPtr;
   char                 *cPtr;
} uPointer_t;

typedef union
{
   long long ll;
   unsigned long long ull;
   double    d;
   void *vp;
   unsigned char ch[8];
} convert_64_t;


// -------------------------- GLOBAL SECTION ---------------------------------
int   ctxHandle = -1;                                                   // context handle. (-1 is invalid).
int   sEventCount=0;

//--------------------------------------------------------------------
// Timing of routines and blocks. Typical usage;
// gettimeofday(&t1, NULL);                  // starting point.
// ... some code to execute ...
// gettimeofday(&t2, NULL);                  // finished timing.
// fprintf(stderr, "routine took %li uS.\n", // report time.
//                       (mConvertUsec(t2)-mConvertUsec(t1)));
//--------------------------------------------------------------------
#define EVENTREADS 100 /* Number of times to read the event. */
#define mConvertUsec(timeval_) ((double) (timeval_.tv_sec*1000000+timeval_.tv_usec))     /* avoid typos, make it a double. */
static struct timeval t1, t2;                                           // used in timing routines to measure performance.

#define _prog_fprintf if (0) fprintf                                    /* change to 1 to enable printing of progress debug messages. TURN OFF if benchmark timing.    */
#define _time_fprintf if (0) fprintf                                    /* change to 1 to enable printing of performance timings.     TURN OFF if benchmark timing.    */

static void* dllib1 = NULL;                                             // Our dynamic library. 

//-----------------------------------------------------------------------------
// Using weak symbols (global declared without a value, so it defers to any
// other global declared in another file WITH a value) allows PAPI to be built
// with the component, but PAPI can still be installed in a system without the
// required library.
//-----------------------------------------------------------------------------

void (*_dl_non_dynamic_init)(void) __attribute__((weak));               // declare a weak dynamic-library init routine pointer.

// ------------------------ LINK DYNAMIC LIBRARIES ---------------------------
// Function pointers for the PCP lib; begin with pm (performance monitor).
// Commented out functions are ones we do not use; so we don't waste space.
static int     (*pmLookupName_ptr)     (int numpid, char **namelist,pmID *pmidlist);
static char*   (*pmErrStr_ptr)         (int code);
// static int     (*pmTraversePMNS_ptr)   (const char *name, void(*func)(const char *));
void           (*pmFreeResult_ptr)     (pmResult *result);
static int     (*pmNewContext_ptr)     (int type, const char *name);
static int     (*pmDestroyContext_ptr) (int handle);
static int     (*pmFetch_ptr)          (int numpid, pmID *pmidlist, pmResult **result);
// static int     (*pmLookupDesc_ptr)     (pmID pmid, pmDesc *desc);
// static int     (*pmGetInDom_ptr)       (pmInDom indom, int **instlist, char ***namelist);
// static int     (*pmLookupText_ptr)     (pmID pmid, int level, char **buffer);
// static char *  (*pmUnitsStr_r_ptr)     (const pmUnits *pu, char *buf, int buflen); 

// -------------------- LOCAL WRAPPERS FOR LIB FUNCTIONS ---------------------
static int     pcp_pmLookupName (int numpid, char **namelist, pmID *pmidlist) 
                  { return ((*pmLookupName_ptr) (numpid, namelist, pmidlist)); }

static char*   pcp_pmErrStr (int code) 
                  { return ((*pmErrStr_ptr) (code)); }

// static int     pcp_pmTraversePMNS (const char *name, void(*func)(const char *)) 
//                   { return ((*pmTraversePMNS_ptr) (name, func)); }

void           pcp_pmFreeResult (pmResult *result) 
                  { return ((*pmFreeResult_ptr) (result)); }

static int     pcp_pmNewContext (int type, const char *name) 
                  { return ((*pmNewContext_ptr) (type,name)); }

static int     pcp_pmDestroyContext(int handle) 
                  { return ((*pmDestroyContext_ptr) (handle));}

static int     pcp_pmFetch (int numpid, pmID *pmidlist, pmResult **result) 
                  { return ((*pmFetch_ptr) (numpid,pmidlist,result)); }

// static int     pcp_pmLookupDesc (pmID pmid, pmDesc *desc) 
//                   { return ((*pmLookupDesc_ptr) (pmid,desc)); }

// static int     pcp_pmGetInDom (pmInDom indom, int **instlist, char ***namelist) 
//                  { return ((*pmGetInDom_ptr) (indom,instlist,namelist)); }

// static int     pcp_pmLookupText(pmID pmid, int level, char **buffer) 
//                   { return ((*pmLookupText_ptr) (pmid, level, buffer)); }

// static char*   pcp_pmUnitsStr_r (const pmUnits *pu, char *buf, int buflen) 
//                   {return ((*pmUnitsStr_r_ptr) (pu, buf, buflen)); }


//-----------------------------------------------------------------------------
// Get all needed function pointers from the Dynamic Link Library. 
//-----------------------------------------------------------------------------

// MACRO checks for Dynamic Lib failure, reports, returns Not Supported.
#define mCheck_DL_Status( err, str )                                          \
   if( err )                                                                  \
   {                                                                          \
      fprintf(stderr, str);                                                   \
      return(-1);                                                             \
   }

// keys for above: Init, InitThrd, InitCtlSt, Stop, ShutdownThrd, ShutdownCmp, Start,
// UpdateCtl, Read, Ctl, SetDom, Reset, Enum, EnumFirst, EnumNext, EnumUmasks, 
// NameToCode, CodeToName, CodeToDesc, CodeToInfo.

// Simplify routine below; relies on ptr names being same as func tags.
#define STRINGIFY(x) #x 
#define TOSTRING(x) STRINGIFY(x)
#define mGet_DL_FPtr(Name)                                                \
   Name##_ptr = dlsym(dllib1, TOSTRING(Name));                            \
   mCheck_DL_Status(dlerror()!=NULL, "PCP library function "              \
                  TOSTRING(Name) " not found in " MYPCPLIB ".");

int _local_linkDynamicLibraries(void) 
{
   if ( _dl_non_dynamic_init != NULL ) {  // If weak var present, statically linked insted of dynamic.
       fprintf(stderr, "This program REQUIRES dynamic linking capabilities.\n");
       exit(-1); 
   }

   dllib1 = dlopen(MYPCPLIB, RTLD_NOW | RTLD_GLOBAL);    // Open lib. MYPCPLIB is defined macro above.
   if (!dllib1) {
       fprintf(stderr, "Component Library '%s' was not found.\n", MYPCPLIB);
       exit(-1); 
   }

//-----------------------------------------------------------------------------
// Collect pointers for routines in shared library.  All below will abort this
// routine with -1, the routine is not found in the dynamic library.
//-----------------------------------------------------------------------------

   mGet_DL_FPtr(pmLookupName);
   mGet_DL_FPtr(pmErrStr);
// mGet_DL_FPtr(pmTraversePMNS);
   mGet_DL_FPtr(pmFreeResult);
   mGet_DL_FPtr(pmNewContext);
   mGet_DL_FPtr(pmDestroyContext);
   mGet_DL_FPtr(pmFetch);
// mGet_DL_FPtr(pmLookupDesc);
// mGet_DL_FPtr(pmGetInDom);
// mGet_DL_FPtr(pmLookupText);
// mGet_DL_FPtr(pmUnitsStr_r);
   return 0;         // If we get here, all above succeeded. 
} // end routine.


//-----------------------------------------------------------------------------
// cbPopulateNameOnly: This is a callback routine, called by pmTraversePMNS.  That
// routine iterates through the PM name space and calls this routine once per
// name. We increment sEventCount as we go, this will be the final count of valid
// array entries. sEventInfoSize will be >= sEventCount.
// WARNING: May realloc() pcp_event_info[], invalidating pointers into it.
//-----------------------------------------------------------------------------

void cbPopulateNameOnly(const char *name) 
{
   (void) name;                                                      // Prevent unused variable warning.
} // end routine.


//-----------------------------------------------------------------------------
// Helper routine, returns a ull value from a value set pointer. Automatically
// does conversions from 32 bit to 64 bit (int32, uint32, fp32).  
//-----------------------------------------------------------------------------
unsigned long long getULLValue(pmValueSet *vset, int value_index) 
{
   unsigned long long value;                                         // our return value.
   convert_64_t convert;                                             // union for conversion.
   uPointer_t myPtr;                                                 // a union helper to avoid warnings.

   if (vset->valfmt == PM_VAL_INSITU) {                              // If the value format is in situ; a 32 bit value.
      convert.ll = vset->vlist[value_index].value.lval;              // .. we can just collect the value immediately.
      value = convert.ull;                                           // .. 
   } else {                                                          // If it is either static or dynamic alloc table,
      pmValueBlock *pmvb = vset->vlist[value_index].value.pval;      // .. value given is a pointer to value block.
      myPtr.cPtr = pmvb->vbuf;                                       // .. use cPtr because vbuf defined as char[1] in pmValueBlock.
      switch (pmvb->vtype) {                                         // Note we restricted the types in init; these cases should agree.
         case  PM_TYPE_32:       // 32-bit signed integer
            convert.ll = myPtr.iPtr[0];
            value = convert.ull;
            break;

         case  PM_TYPE_U32:      // 32-bit unsigned integer
            value =  myPtr.uiPtr[0];
            break;

         case  PM_TYPE_64:       // 64-bit signed integer
            convert.ll = myPtr.llPtr[0];
            value = convert.ull;
            break;

         case  PM_TYPE_U64:      // 64-bit unsigned integer
            value = myPtr.ullPtr[0];
            break;

         case  PM_TYPE_FLOAT:    // 32-bit floating point
            convert.d = myPtr.fPtr[0];  // convert first.
            value = convert.ull;
            break;

         case  PM_TYPE_DOUBLE:   // 64-bit floating point
            convert.d = myPtr.dPtr[0];
            value = convert.ull;
            break;

         case  PM_TYPE_STRING:   // array of char 
            convert.vp = myPtr.cPtr;
            value = convert.ull;            
            break;

         default:
            fprintf(stderr, "%s:%i pmValueBlock (from PCP) contains an unrecognized value type=%i.\n",
               __FILE__, __LINE__,  pmvb->vtype);
            convert.ll = -1;                                            // A flag besides zero
            value = convert.ull;
      } // end switch on type. 
   } // if pmValueBlock value.

   return(value);                                                       // exit with result.
} // end routine. 


//----------------------------------------------------------------------------
// main(). intialize the lib, then work on reading the value. 
//---------------------------------------------------------------------------
int main (int argc, char **argv) 
{
   (void) argc; (void) argv;                                            // Prevent not used warning.
   #define hostnameLen 512 /* constant used multiple times. */
   char hostname[hostnameLen];                                          // host name.
   char *pcpName;                                                       // name of event to collect. 
   int  i, j, ret;

   if (argc == 1) {                                                     // If no arguments given,
      printf("Initialize, uS 1st PCP Read, uS %i PCP Reads\n", EVENTREADS);    // OUTPUT Header for CSV.
      return 0;                                                         // done.
   }

   if (argc != 2) {
      fprintf(stderr, "%s:%i ERROR Invalid number of arguments; must be 0 or 1.\n", __FILE__, __LINE__); // report.
      exit(-1);
   }

   // Get args.
   pcpName  = argv[1];                                                  // collect the pcp event name.
   sEventCount = 1;                                                     // currently we only allow one name.
   char **allNames=calloc(sEventCount, sizeof(char*));                  // Make an array for all names. 

   // Modify here if you have more than one name.
   allNames[0] = pcpName;                                               // Just this one, so far!

   // Make array to look up PMIDs from the names above.
   pmID *allPMID=calloc(sEventCount, sizeof(pmID));                     // Make an array for results.


   //-------------------------------------------------------------------
   // Begin initialization timing.
   //-------------------------------------------------------------------

   gettimeofday(&t1, NULL);
   ret = _local_linkDynamicLibraries();
   if ( ret != 0) {                                                     // Failure to get lib.
      fprintf(stderr, "Failed attempt to link to PCP "
              "library '%s'.\n", MYPCPLIB);
      exit(-1); 
   }

   _prog_fprintf(stderr, "Linked to %s.\n", MYPCPLIB);                  // debug only; turn off if timing.

   ret = gethostname(hostname, hostnameLen);                            // Try to get the host hame.
   if( gethostname(hostname, hostnameLen) != 0) {                       // If we can't get the hostname, 
      fprintf(stderr, "Failed system call, gethostname() "
            "returned %i.", ret);
      exit(-1);
   }
   #undef hostnameLen /* done with it. */

   _prog_fprintf(stderr, "%s:%i retrieved hostname='%s'\n", __FILE__, __LINE__, hostname); // show progress.

   ctxHandle = pcp_pmNewContext(PM_CONTEXT_HOST, hostname);             // Set the new context to hostname retrieved.
   if (ctxHandle < 0) {
      fprintf(stderr, "Cannot connect to PM Daemon on host \"%s\".\n "
         "(Ensure this machine has Performance Co-Pilot installed.)\n", hostname);
     exit(-1); 
   }

   _prog_fprintf(stderr, "%s:%i Found ctxHandle=%i\n", __FILE__, __LINE__, ctxHandle); // show progress.

   //-------------------------------------------------------------------
   // Fetch some events, in a loop. We make an array of names and an 
   // array of PMIDs, we need the PMIDs to fetch values. Currently we
   // only do one for a benchmark; but this is written to do more than
   // one if desired. 
   //-------------------------------------------------------------------
   // We will time this for cold-start overhead, but separately;
   // PMID need only be found once.

   ret = pcp_pmLookupName(sEventCount, allNames, allPMID);              // .. Get a block of PMIDs for a block of names.
   if (ret < 0) {                                                       // .. Failure...
      fprintf(stderr, "pmLookupName for %i names failed; ret=%i [%s].\n", 
         sEventCount, ret, pcp_pmErrStr(ret));
      exit(-1);
   }

   gettimeofday(&t2, NULL);
   printf("%9.1f,", (mConvertUsec(t2)-mConvertUsec(t1)));               // OUTPUT INIT TIME.

   // Time a SINGLE pmFetch now. 
   pmResult *allFetch = NULL;                                           // result of pmFetch. 
   gettimeofday(&t1, NULL);
   ret = pcp_pmFetch(sEventCount, allPMID, &allFetch);                  // Fetch (read) all the events.
   if( ret != 0) {                                                      // If fetch failed .. 
      printf("ERROR pcp_pmFetch FAILED.\n");
      fprintf(stderr, "%s:%i pcp_pmFetch failed, return=%i.\n", 
         __FILE__, __LINE__, ret);                                      // .. report error.
      free(allNames);                                                   // .. Locals allocations not needed anymore.
      free(allPMID);                                                    // .. the pmIDs we read.
      if (allFetch != NULL) pcp_pmFreeResult(allFetch);                 // .. release the results we fetched.
      exit(-1);                                                         // .. Blow up.
   }

   // The fetch worked.
   for (i=0; i<sEventCount; i++) {                                      // for each PMID we fetched,
      pmValueSet *vset = allFetch->vset[i];                             // get the result for its event.
      retZero(getULLValue(vset, 0));                                    // .. .. translate as needed. Don't optimize out.
   }

   pcp_pmFreeResult(allFetch);                                          // Free it up.

   gettimeofday(&t2, NULL);
   printf("%9.1f,",  (mConvertUsec(t2)-mConvertUsec(t1)));              // OUTPUT the First Read. 

   // Time EVENTREADS pmFetches in a row, with cleanups.
   gettimeofday(&t1, NULL);
   for (j=0; j<EVENTREADS; j++) {
      ret = pcp_pmFetch(sEventCount, allPMID, &allFetch);               // Fetch (read) all the events.
      if( ret != 0) {                                                   // If fetch failed .. 
         printf("ERROR pcp_pmFetch in loop %i FAILED.\n", j);
         fprintf(stderr, "%s:%i pcp_pmFetch failed, return=%i.\n", 
            __FILE__, __LINE__, ret);                                   // .. report error.
         free(allNames);                                                // .. Locals allocations not needed anymore.
         free(allPMID);                                                 // .. the pmIDs we read.
         if (allFetch != NULL) pcp_pmFreeResult(allFetch);              // .. release the results we fetched.
         exit(-1);                                                      // .. Blow up.
      }

      // The fetch worked.
      for (i=0; i<sEventCount; i++) {                                   // for each PMID we fetched,
         pmValueSet *vset = allFetch->vset[i];                          // get the result for its event.
         retZero(getULLValue(vset, 0));                                 // .. .. translate as needed. Don't optimize out.
      }

      pcp_pmFreeResult(allFetch);                                       // Free it up.
   } // end 100 loop.

   gettimeofday(&t2, NULL);
   printf("%9.1f\n", (mConvertUsec(t2)-mConvertUsec(t1)));              // OUTPUT the repeated time.

   //-------------------------------------------------------------------
   // Cleanup, and shutdown PCP connection. 
   //-------------------------------------------------------------------

   free(allNames);                                                      // Locals allocations not needed anymore.
   free(allPMID);                                                       // .. the pmIDs we read.
   pcp_pmDestroyContext(ctxHandle);                                     // context handle; fails to free malloced memory, says valgrind.
   return 0;
} // end MAIN routine.



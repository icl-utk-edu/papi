/*
 * Basic PAPI Test for BG/P
 *
 *  NOTE:  If a PAPI function is not listed below, the function is
 *         untested and user beware...
 *
 * The following high level functions are called...
 *   PAPI_num_counters  - get the number of hardware counters available on the system
 *   PAPI_flips  - simplified call to get Mflips/s (floating point instruction rate), real and processor time
 *   PAPI_flops  - simplified call to get Mflops/s (floating point operation rate), real and processor time
 *   PAPI_ipc  - gets instructions per cycle, real and processor time
 *   PAPI_accum_counters  - add current counts to array and reset counters
 *   PAPI_read_counters  - copy current counts to array and reset counters
 *   PAPI_start_counters  - start counting hardware events
 *   PAPI_stop_counters  - stop counters and return current counts
 *
 * The following low level functions are called...
 *   PAPI_accum  - accumulate and reset hardware events from an event set
 *   PAPI_add_event  - add single PAPI preset or native hardware event to an event set
 *   PAPI_cleanup_eventset  - remove all PAPI events from an event set
 *   PAPI_create_eventset  - create a new empty PAPI event set
 *   PAPI_destroy_eventset  - deallocates memory associated with an empty PAPI event set
 *   PAPI_enum_event  - return the event code for the next available preset or natvie event
 *   PAPI_event_code_to_name  - translate an integer PAPI event code into an ASCII PAPI preset or native name
 *   PAPI_event_name_to_code  - translate an ASCII PAPI preset or native name into an integer PAPI event code
 *   PAPI_get_dmem_info  - get dynamic memory usage information
 *   PAPI_get_event_info  - get the name and descriptions for a given preset or native event code
 *   PAPI_get_executable_info  - get the executable’s address space information
 *   PAPIF_get_exe_info  - Fortran version of PAPI_get_executable_info with different calling semantics
 *   PAPI_get_hardware_info  - get information about the system hardware
 *   PAPI_get_multiplex  - get the multiplexing status of specified event set
 *   PAPI_get_real_cyc  - return the total number of cycles since some arbitrary starting point
 *   PAPI_get_real_usec  - return the total number of microseconds since some arbitrary starting point
 *   PAPI_get_shared_lib_info  - get information about the shared libraries used by the process
 *   PAPI_get_virt_cyc  - return the process cycles since some arbitrary starting point
 *   PAPI_get_virt_usec  - return the process microseconds since some arbitrary starting point
 *   PAPI_is_initialized  - return the initialized state of the PAPI library
 *   PAPI_library_init  - initialize the PAPI library
 *   PAPI_list_events  - list the events that are members of an event set
 *   PAPI_num_hwctrs  - return the number of hardware counters
 *   PAPI_num_events  - return the number of events in an event set
 *   PAPI_query_event  - query if a PAPI event exists
 *   PAPI_read  - read hardware events from an event set with no reset
 *   PAPI_remove_event  - remove a hardware event from a PAPI event set
 *   PAPI_reset  - reset the hardware event counts in an event set
 *   PAPI_shutdown  - finish using PAPI and free all related resources
 *   PAPI_start  - start counting hardware events in an event set
 *   PAPI_state  - return the counting state of an event set
 *   PAPI_stop  - stop counting hardware events in an event set and return current events
 *   PAPI_write  - write counter values into counters
 *     NOTE:  Not supported when UPC is running, and when not running, only changes local PAPI memory.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <common/alignment.h>

#include <spi/bgp_SPI.h>
#include "papiStdEventDefs.h"
#include "papi.h"
#include "linux-bgp-native-events.h"

#define MAX_COUNTERS 256
#define NUMBER_COUNTERS_PER_ROW 8
/*
 * Prototypes...
 */
void Do_Tests(void);
void Do_Low_Level_Tests(void);
void Do_High_Level_Tests(void);
void Do_Multiplex_Tests(void);
void Run_Cycle(const int pNumEvents);
void Zero_Local_Counters(long long* pCounters);
void FPUArith(void);
void List_PAPI_Events(const int pEventSet, int* pEvents, int* xNumEvents);
void Print_Native_Counters();
void Print_Native_Counters_via_Buffer(const BGP_UPC_Read_Counters_Struct_t* pBuffer);
void Print_Native_Counters_for_PAPI_Counters(const int pEventSet);
void Print_Native_Counters_for_PAPI_Counters_From_List(const int* pEvents, const int pNumEvents);
void Print_PAPI_Counters(const int pEventSet, const long long* pCounters);
void Print_PAPI_Counters_From_List(const int* pEventList, const int pNumEvents, const long long* pCounters);
void Print_Counters(const int pEventSet);
void Print_Node_Info(void);
void Read_Native_Counters(const int pLength);
void Print_PAPI_Events(const int pEventSet);
void Print_Counter_Values(const long long* pCounters, const int pNumCounters);
void DumpInHex(const char* pBuffer, int pSize);


/*
 * Global variables...
 */
int PAPI_Events[MAX_COUNTERS];
long long PAPI_Counters[MAX_COUNTERS];
char Native_Buffer[BGP_UPC_MAXIMUM_LENGTH_READ_COUNTERS_STRUCTURE];
double x[32] ALIGN_L3_CACHE;


const int NumEventsPerSet = MAX_COUNTERS;
const int MaxPresetEventId = 104;
const int MaxNativeEventId = 511;

int main(int argc, char * argv[]) {
  _BGP_Personality_t personality;
  int pRank=0, pMode=-2, pCore=0, pEdge=1, xActiveCore=0, xActiveRank=0, xRC;

  /*
   * Check args, print test inputs.
   */

  if ( argc > 1 )
    sscanf(argv[1], "%d", &pRank);
  if ( argc > 2 )
    sscanf(argv[2], "%d", &pMode);
  if ( argc > 3 )
    sscanf(argv[3], "%d", &pCore);
  if ( argc > 4 )
    sscanf(argv[4], "%d", &pEdge);

/*
 * Check for valid rank...
 */
  if ( pRank < 0 || pRank > 31 ) {
    printf("Invalid rank (%d) specified\n", pRank);
    exit(1);
  }
/*
 * Check for valid mode...
 * Mode = -2 means use what was initialized by CNK
 * Mode = -1 means to initialize with the default
 * Mode = 0-3 means to initialize with mode 0-3
 */
  if ( pMode < -2 || pMode > 3 ) {
    printf("Invalid mode (%d) specified\n", pMode);
    exit(1);
  }
/*
 * Check for valid core...
 */
  if ( pCore < 0 || pCore > 3 ) {
    printf("Invalid core (%d) specified\n", pCore);
    exit(1);
  }
/*
 * Check for valid edge...
 * Edge = 1  means initialize with the default edge
 * Edge = 0  means initialize with level high
 * Edge = 4  means initialize with edge rise
 * Edge = 8  means initialize with edge fall
 * Edge = 12 means initialize with level low
 */
  if ( pEdge != 0 && pEdge != 1 && pEdge != 4 && pEdge != 8 && pEdge != 12 ) {
    printf("Invalid edge (%d) specified\n", pEdge);
    exit(1);
  }

/*
 * Initialize the UPC environment...
 * NOTE:  Must do this from all 'ranks'...
 */
//  BGP_UPC_Initialize();
  xRC = PAPI_library_init(PAPI_VER_CURRENT);
  if (xRC != 50921472) {
    printf("PAPI_library_init failed:  xRC=%d, ending...\n", xRC);
    exit(1);
  }

/*
 * Only run if this is specified rank...
 */

  xRC = Kernel_GetPersonality(&personality, sizeof(_BGP_Personality_t));
  if (xRC !=0) {
    printf(" Kernel_GetPersonality returned %d\n",xRC) ;
    exit(xRC);
  }
  xActiveRank = personality.Network_Config.Rank;
  xActiveCore = Kernel_PhysicalProcessorID();

  printf("Rank %d, core %d reporting...\n", xActiveRank, xActiveCore);

  if (xActiveRank != pRank) {
    printf("Rank %d is not to run...  Exiting...\n", xActiveRank);
    exit(0);
  }

  if ( xActiveCore == pCore ) {
    printf("Program is to run on rank %d core %d, using mode= %d, edge= %d\n", pRank, xActiveCore, pMode, pEdge);
  }
  else {
    printf("Program is NOT to run on rank %d core %d...  Exiting...\n", pRank, xActiveCore);
    exit(0);
  }

/*
 * Main processing...
 */
  printf("************************************************************\n");
  printf("* Configuration parameters used:                           *\n");
  printf("*   Rank = %d                                              *\n", pRank);
  printf("*   Mode = %d                                              *\n", pMode);
  printf("*   Core = %d                                              *\n", pCore);
  printf("*   Edge = %d                                              *\n", pEdge);
  printf("************************************************************\n\n");

  printf("Print config after PAPI_library_init...\n");
  BGP_UPC_Print_Config();

/*
 * If we are to initialize, do so with user mode and edge...
 * Otherwise, use what was initialized by CNK...
 */
  if (pMode > -2) {
    BGP_UPC_Initialize_Counter_Config(pMode, pEdge);
    printf("UPC unit(s) initialized with mode=%d, edge=%d...\n", pMode, pEdge);
  }

  printf("Before running the main test procedure...\n");
  BGP_UPC_Print_Config();
  BGP_UPC_Print_Counter_Values(BGP_UPC_READ_EXCLUSIVE);

/*
 * Perform the main test procedure...
 */
  Do_Tests();

/*
 * Print out final configuration and results...
 */
  printf("After running the main test procedure...\n");
  BGP_UPC_Print_Config();
  BGP_UPC_Print_Counter_Values(BGP_UPC_READ_EXCLUSIVE);

  exit(0);
}


/*
 * Do_Tests
 */

void Do_Tests(void) {
  printf("==>  Do_Tests():  Beginning of the main body...\n");

  //  NOTE:  PAPI_library_init() has already been done for each participating node
  //         prior to calling this routine...

  Do_Low_Level_Tests();
  Do_High_Level_Tests();
  Do_Multiplex_Tests(); // NOTE:  Not supported...
  PAPI_shutdown();

  printf("==>  Do_Tests():  End of the main body...\n");
  fflush(stdout);

  return;
}

/*
 * Do_Low_Level_Tests
 */

void Do_Low_Level_Tests(void) {
  int xRC, xEventSet, xEventCode, xState;
  long long xLLValue;
  char xName[256];

  printf("==>  Do_Low_Level_Tests():  Beginning of the main body...\n");

  /*
   * Low-level API tests...
   */

  xRC = PAPI_is_initialized();
  if (xRC == 1)
    printf("SUCCESS:  PAPI has been low-level initialized by main()...\n");
  else {
    printf("FAILURE:  PAPI has not been properly initialized by main(), xRC=%d, ending...\n", xRC);
    return;
  }

  /*
   * Print out the node information with respect to UPC units...
   */
  Print_Node_Info();

  /*
   * Zero the buffers for counters...
   */
  Zero_Local_Counters(PAPI_Counters);
  BGP_UPC_Read_Counters_Struct_t* xTemp;
  xTemp = (BGP_UPC_Read_Counters_Struct_t*)(void*)Native_Buffer;
  Zero_Local_Counters(xTemp->counter);

  /*
   * Start of real tests...
   */
  xLLValue = -1;
  xLLValue = PAPI_get_real_cyc();
  printf("PAPI_get_real_cyc:  xLLValue=%lld...\n", xLLValue);

  xLLValue = -1;
  xLLValue = PAPI_get_virt_cyc();
  printf("PAPI_get_virt_cyc:  xLLValue=%lld...\n", xLLValue);

  xLLValue = -1;
  xLLValue = PAPI_get_real_usec();
  printf("PAPI_get_real_usec:  xLLValue=%lld...\n", xLLValue);

  xLLValue = -1;
  xLLValue = PAPI_get_virt_usec();
  printf("PAPI_get_virt_usec:  xLLValue=%lld...\n", xLLValue);

  xRC = PAPI_num_hwctrs();
  if (xRC == 256)
    printf("SUCCESS:  PAPI_num_hwctrs returned 256 hardware counters...\n");
  else
    printf("FAILURE:  PAPI_num_hwctrs failed, returned xRC=%d...\n", xRC);

  *xName = 0;
  char* xEventName_1 = "PAPI_L3_LDM";
  xRC = PAPI_event_code_to_name(PAPI_L3_LDM, xName);
  if (xRC == PAPI_OK) {
    xRC = strcmp(xName,xEventName_1);
    if (!xRC)
      printf("SUCCESS:  PAPI_event_code_to_name for PAPI_L3_LDM...\n");
    else
      printf("FAILURE:  PAPI_event_code_to_name returned incorrect name, xName=%s\n", xName);
  }
  else
    printf("FAILURE:  PAPI_event_code_to_name failed, xRC=%d...\n", xRC);

  *xName = 0;
  char* xEventName_2 = "PNE_BGP_PU1_IPIPE_INSTRUCTIONS";
  xRC = PAPI_event_code_to_name(PNE_BGP_PU1_IPIPE_INSTRUCTIONS, xName);
  if (xRC == PAPI_OK) {
    xRC = strcmp(xName,xEventName_2);
    if (!xRC)
      printf("SUCCESS:  PAPI_event_code_to_name for PNE_BGP_PU1_IPIPE_INSTRUCTIONS...\n");
    else
      printf("FAILURE:  PAPI_event_code_to_name returned incorrect name, xName=%s\n", xName);
    }
  else
    printf("FAILURE:  PAPI_event_code_to_name failed, xRC=%d...\n", xRC);

  strcpy(xName,"PAPI_L3_LDM");
  xRC = PAPI_event_name_to_code(xName, &xEventCode);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x8000000E)
      printf("SUCCESS:  PAPI_event_name_to_code for PAPI_L3_LDM...\n");
    else
      printf("FAILURE:  PAPI_event_name_to_code returned incorrect code, xEventCode=%d\n", xEventCode);
  else
    printf("FAILURE:  PAPI_event_name_to_code failed, xRC=%d...\n", xRC);

  strcpy(xName,"PNE_BGP_PU1_IPIPE_INSTRUCTIONS");
  xRC = PAPI_event_name_to_code(xName, &xEventCode);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x40000027)
      printf("SUCCESS:  PAPI_event_name_to_code for PNE_BGP_PU1_IPIPE_INSTRUCTIONS...\n");
    else
      printf("FAILURE:  PAPI_event_name_to_code returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_event_name_to_code failed, xRC=%d...\n", xRC);

  xEventCode = 0x80000000;
  xRC = PAPI_enum_event(&xEventCode, PAPI_ENUM_ALL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x80000001)
      printf("SUCCESS:  PAPI_enum_event for 0x80000000 PAPI_PRESET_ENUM_ALL, returned 0x80000001...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x80000000 PAPI_PRESET_ENUM_ALL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x80000000 PAPI_PRESET_ENUM_ALL failed, xRC=%d...\n", xRC);

  xEventCode = 0x80000002;
  xRC = PAPI_enum_event(&xEventCode, PAPI_ENUM_ALL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x80000003)
      printf("SUCCESS:  PAPI_enum_event for 0x80000002 PAPI_PRESET_ENUM_ALL, returned 0x80000003...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x80000002 PAPI_PRESET_ENUM_ALL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x80000002 PAPI_PRESET_ENUM_ALL failed, xRC=%d...\n", xRC);

  xEventCode = 0x80000067;
  xRC = PAPI_enum_event(&xEventCode, PAPI_ENUM_ALL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x80000068)
      printf("SUCCESS:  PAPI_enum_event for 0x80000067 PAPI_PRESET_ENUM_ALL, returned 0x80000068...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x80000067 PAPI_PRESET_ENUM_ALL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x80000067 PAPI_PRESET_ENUM_ALL failed, xRC=%d...\n", xRC);

  xEventCode = 0x80000068;
  xRC = PAPI_enum_event(&xEventCode, PAPI_ENUM_ALL);
  if (xRC == PAPI_ENOEVNT)
    printf("SUCCESS:  PAPI_enum_event for 0x80000068 PAPI_PRESET_ENUM_ALL, no next event...\n");
  else
    printf("FAILURE:  PAPI_enum_event for 0x80000068 PAPI_PRESET_ENUM_ALL failed, xRC=%d...\n", xRC);

  xEventCode = 0x40000000;
  xRC = PAPI_enum_event(&xEventCode, PAPI_ENUM_ALL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x40000001)
      printf("SUCCESS:  PAPI_enum_event for 0x40000000 PAPI_PRESET_ENUM_ALL, returned 0x40000001...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x40000000 PAPI_PRESET_ENUM_ALL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x40000000 PAPI_PRESET_ENUM_ALL failed, xRC=%d...\n", xRC);

  xEventCode = 0x40000001;
  xRC = PAPI_enum_event(&xEventCode, PAPI_ENUM_ALL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x40000002)
      printf("SUCCESS:  PAPI_enum_event for 0x40000001 PAPI_PRESET_ENUM_ALL, returned 0x40000002...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x40000001 PAPI_PRESET_ENUM_ALL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x40000001 PAPI_PRESET_ENUM_ALL failed, xRC=%d...\n", xRC);

  xEventCode = 0x400000FC;
  xRC = PAPI_enum_event(&xEventCode, PAPI_ENUM_ALL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x400000FF)
      printf("SUCCESS:  PAPI_enum_event for 0x400000FC PAPI_PRESET_ENUM_ALL, returned 0x400000FF...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x400000FC PAPI_PRESET_ENUM_ALL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x400000FC PAPI_PRESET_ENUM_ALL failed, xRC=%d...\n", xRC);

  xEventCode = 0x400001FD;
  xRC = PAPI_enum_event(&xEventCode, PAPI_ENUM_ALL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x400001FF)
      printf("SUCCESS:  PAPI_enum_event for 0x400001FD PAPI_ENUM_ALL, returned 0x400001FF...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x400001FD PAPI_ENUM_ALL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x400001FD PAPI_ENUM_ALL failed, xRC=%d...\n", xRC);

  xEventCode = 0x400001FF;
  xRC = PAPI_enum_event(&xEventCode, PAPI_ENUM_ALL);
  if (xRC == PAPI_ENOEVNT)
    printf("SUCCESS:  PAPI_enum_event for 0x400001FF PAPI_PRESET_ENUM_ALL, no next event...\n");
  else
    printf("FAILURE:  PAPI_enum_event for 0x400001FF PAPI_PRESET_ENUM_ALL failed, xRC=%d...\n", xRC);

  xEventCode = 0x80000000;
  xRC = PAPI_enum_event(&xEventCode, PAPI_PRESET_ENUM_AVAIL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x80000001)
      printf("SUCCESS:  PAPI_enum_event for 0x80000000 PAPI_PRESET_ENUM_AVAIL, returned 0x80000001...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x80000000PAPI_PRESET_ENUM_AVAIL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x80000000PAPI_PRESET_ENUM_AVAIL failed, xRC=%d...\n", xRC);

  xEventCode = 0x80000002;
  xRC = PAPI_enum_event(&xEventCode, PAPI_PRESET_ENUM_AVAIL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x80000006)
      printf("SUCCESS:  PAPI_enum_event for 0x80000002 PAPI_PRESET_ENUM_AVAIL, returned 0x80000006...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x80000002 PAPI_PRESET_ENUM_AVAIL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x80000002 PAPI_PRESET_ENUM_AVAIL failed, xRC=%d...\n", xRC);

  xEventCode = 0x80000067;
  xRC = PAPI_enum_event(&xEventCode, PAPI_PRESET_ENUM_AVAIL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x80000068)
      printf("SUCCESS:  PAPI_enum_event for 0x80000067 PAPI_PRESET_ENUM_AVAIL, returned 0x80000068...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x80000067 PAPI_PRESET_ENUM_AVAIL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x80000067 PAPI_PRESET_ENUM_AVAIL failed, xRC=%d...\n", xRC);

  xEventCode = 0x80000068;
  xRC = PAPI_enum_event(&xEventCode, PAPI_PRESET_ENUM_AVAIL);
  if (xRC == PAPI_ENOEVNT)
    printf("SUCCESS:  PAPI_enum_event for 0x80000068 PAPI_PRESET_ENUM_AVAIL, no next event...\n");
  else
    printf("FAILURE:  PAPI_enum_event for 0x80000068 PAPI_PRESET_ENUM_AVAIL failed, xRC=%d...\n", xRC);

  xEventCode = 0x40000000;
  xRC = PAPI_enum_event(&xEventCode, PAPI_PRESET_ENUM_AVAIL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x40000001)
      printf("SUCCESS:  PAPI_enum_event for 0x40000000 PAPI_PRESET_ENUM_AVAIL, returned 0x40000001...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x40000000 PAPI_PRESET_ENUM_AVAIL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x40000000 PAPI_PRESET_ENUM_AVAIL failed, xRC=%d...\n", xRC);

  xEventCode = 0x40000001;
  xRC = PAPI_enum_event(&xEventCode, PAPI_PRESET_ENUM_AVAIL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x40000002)
      printf("SUCCESS:  PAPI_enum_event for 0x40000001 PAPI_PRESET_ENUM_AVAIL, returned 0x40000002...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x40000001 PAPI_PRESET_ENUM_AVAIL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x40000001 PAPI_PRESET_ENUM_AVAIL failed, xRC=%d...\n", xRC);

  printf("NOTE:  Might get two messages indicating invalid event id specified for 253 and 254.  These are OK...\n");
  xEventCode = 0x400000FC;
  xRC = PAPI_enum_event(&xEventCode, PAPI_PRESET_ENUM_AVAIL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x400000FF)
      printf("SUCCESS:  PAPI_enum_event for 0x400000FC PAPI_PRESET_ENUM_AVAIL, returned 0x400000FF...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x400000FC PAPI_PRESET_ENUM_AVAIL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x400000FC PAPI_PRESET_ENUM_AVAIL failed, xRC=%d...\n", xRC);

  printf("NOTE:  Might get one message indicating invalid event id specified for 510.  This is OK...\n");
  xEventCode = 0x400001FD;
  xRC = PAPI_enum_event(&xEventCode, PAPI_PRESET_ENUM_AVAIL);
  if (xRC == PAPI_OK)
    if (xEventCode == 0x400001FF)
      printf("SUCCESS:  PAPI_enum_event for 0x400001FD PAPI_PRESET_ENUM_AVAIL, returned 0x400001FF...\n");
    else
      printf("FAILURE:  PAPI_enum_event for 0x400001FD PAPI_PRESET_ENUM_AVAIL returned incorrect code, xEventCode=%8.8x\n", xEventCode);
  else
    printf("FAILURE:  PAPI_enum_event for 0x400001FD PAPI_PRESET_ENUM_AVAIL failed, xRC=%d...\n", xRC);

  xEventCode = 0x400001FF;
  xRC = PAPI_enum_event(&xEventCode, PAPI_PRESET_ENUM_AVAIL);
  if (xRC == PAPI_ENOEVNT)
    printf("SUCCESS:  PAPI_enum_event for 0x400001FF PAPI_PRESET_ENUM_AVAIL, no next event...\n");
  else
    printf("FAILURE:  PAPI_enum_event for 0x400001FF PAPI_PRESET_ENUM_AVAIL failed, xRC=%d...\n", xRC);

  PAPI_dmem_info_t xDmemSpace;
  xRC = PAPI_get_dmem_info(&xDmemSpace);
  if (xRC == PAPI_OK) {
    DumpInHex((char*)&xDmemSpace, sizeof( PAPI_dmem_info_t));
    printf("SUCCESS:  PAPI_get_dmem_info...\n");
  }
  else
    printf("FAILURE:  PAPI_get_dmem_info failed, xRC=%d...\n", xRC);

  PAPI_event_info_t xInfoSpace;
  xRC = PAPI_get_event_info(PAPI_L3_LDM, &xInfoSpace);
  if (xRC == PAPI_OK) {
    DumpInHex((char*)&xInfoSpace, sizeof( PAPI_event_info_t));
    printf("SUCCESS:  PAPI_get_event_info for PAPI_L3_LDM...\n");
  }
  else
    printf("FAILURE:  PAPI_get_event_info failed for PAPI_L3_LDM, xRC=%d...\n", xRC);

  const PAPI_exe_info_t* xExeInfo = NULL;
  if ((xExeInfo = PAPI_get_executable_info()) != NULL) {
    DumpInHex((char*)xExeInfo, sizeof( PAPI_exe_info_t));
    printf("SUCCESS:  PAPI_get_executable_info...\n");
  }
  else
    printf("FAILURE:  PAPI_get_executable_info failed, returned null pointer...\n");

  const PAPI_hw_info_t* xHwInfo = NULL;
  if ((xHwInfo = PAPI_get_hardware_info()) != NULL) {
    DumpInHex((char*)xHwInfo, sizeof( PAPI_hw_info_t));
    printf("SUCCESS:  PAPI_get_hardware_info...\n");
  }
  else
    printf("FAILURE:  PAPI_get_hardware_info failed, returned null pointer...\n");

  const PAPI_shlib_info_t* xShLibInfo = NULL;
  if ((xShLibInfo = PAPI_get_shared_lib_info()) != NULL) {
    DumpInHex((char*)xShLibInfo, sizeof( PAPI_shlib_info_t));
    printf("SUCCESS:  PAPI_get_shared_lib_info...\n");
  }
  else
    printf("FAILURE:  PAPI_get_shared_lib_info failed, returned null pointer...\n");

  xEventSet = PAPI_NULL;
  xRC = PAPI_create_eventset(&xEventSet);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_create_eventset created...\n");
  else {
    printf("FAILURE:  PAPI_create_eventset failed, xRC=%d...\n", xRC);
    return;
  }

  printf("==>  No events should be in the event set...\n");
  Print_Counters(xEventSet);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 0)
    printf("SUCCESS:  PAPI_num_events returned 0...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_add_event(xEventSet, PAPI_L1_DCM);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_add_event PAPI_L1_DCM...\n");
  else
    printf("FAILURE:  PAPI_add_event PAPI_L1_DCM failed, xRC=%d...\n", xRC);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 1)
    printf("SUCCESS:  PAPI_num_events returned 1...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_add_event(xEventSet, PNE_BGP_PU3_L2_MEMORY_WRITES);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_add_event PNE_BGP_PU3_L2_MEMORY_WRITES...\n");
  else
    printf("FAILURE:  PAPI_add_event PNE_BGP_PU3_L2_MEMORY_WRITES failed, xRC=%d...\n", xRC);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 2)
    printf("SUCCESS:  PAPI_num_events returned 2...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_add_event(xEventSet, BGP_PU3_L2_MEMORY_WRITES);
  if (xRC == PAPI_EINVAL)
    printf("SUCCESS:  PAPI_add_event BGP_PU3_L2_MEMORY_WRITES not allowed...\n");
  else
    printf("FAILURE:  PAPI_add_event BGP_PU3_L2_MEMORY_WRITES allowed, or failed incorrectly..., xRC=%d...\n", xRC);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 2)
    printf("SUCCESS:  PAPI_num_events returned 2...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_add_event(xEventSet, 0x40000208);
  if (xRC == PAPI_ENOEVNT)
    printf("SUCCESS:  PAPI_add_event 0x40000208 not allowed...\n");
  else
    printf("FAILURE:  PAPI_add_event 0x40000208 allowed, or failed incorrectly..., xRC=%d...\n", xRC);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 2)
    printf("SUCCESS:  PAPI_num_events returned 2...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_add_event(xEventSet, PAPI_L1_ICM);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_add_event PAPI_L1_ICM...\n");
  else
    printf("FAILURE:  PAPI_add_event PAPI_L1_ICM failed, xRC=%d...\n", xRC);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 3)
    printf("SUCCESS:  PAPI_num_events returned 3...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_add_event(xEventSet, PAPI_L1_TCM);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_add_event PAPI_L1_TCM...\n");
  else
    printf("FAILURE:  PAPI_add_event PAPI_L1_TCM failed, xRC=%d...\n", xRC);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 4)
    printf("SUCCESS:  PAPI_num_events returned 4...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_add_event(xEventSet, PAPI_L1_DCM);
  if (xRC == PAPI_ECNFLCT)
    printf("SUCCESS:  PAPI_add_event, redundantly adding PAPI_L1_DCM not allowed...\n");
  else
    printf("FAILURE:  PAPI_add_event PAPI_L1_DCM failed incorrectly, xRC=%d...\n", xRC);

  xRC = PAPI_add_event(xEventSet, PNE_BGP_PU3_L2_MEMORY_WRITES);
  if (xRC == PAPI_ECNFLCT)
    printf("SUCCESS:  PAPI_add_event, redundantly adding PNE_BGP_PU3_L2_MEMORY_WRITES not allowed...\n");
  else
    printf("FAILURE:  PAPI_add_event PNE_BGP_PU3_L2_MEMORY_WRITES failed incorectly, xRC=%d...\n", xRC);

  printf("\n==>  All events added... Perform a read now...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read...\n");
  else
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);

  printf("\n==>  Perform a reset now...\n");
  xRC = PAPI_reset(xEventSet);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_reset...\n");
  else
    printf("FAILURE:  PAPI_reset failed, xRC=%d...\n", xRC);

  printf("\n==>  Perform another read now...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read...\n");
  else
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);

  printf("\n==>  Should be 4 counters below, preset, native, preset, and preset.  All counter values should be zero.\n");
  Print_Counters(xEventSet);

  printf("\n==>  Stop the UPC now...\n");
  xRC = PAPI_stop(xEventSet, PAPI_Counters);
  if (xRC == PAPI_ENOTRUN)
    printf("SUCCESS:  PAPI_stop, but not running...\n");
  else
    printf("FAILURE:  PAPI_stop failed incorectly, xRC=%d...\n", xRC);

  printf("\n==>  Start the UPC now...\n");
  xRC = PAPI_start(xEventSet);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_start...\n");
  else {
    printf("FAILURE:  PAPI_start failed, xRC=%d...\n", xRC);
    return;
  }

  printf("\n==>  Try to start it again...\n");
  xRC = PAPI_start(xEventSet);
  if (xRC == PAPI_EISRUN)
    printf("SUCCESS:  PAPI_start, but already running...\n");
  else
    printf("FAILURE:  PAPI_start failed incorectly, xRC=%d...\n", xRC);

  FPUArith();

  printf("\n==>  Stop the UPC after the arithmetic was performed...  The individual native counter values will be greater than the PAPI counters because the PAPI counters are read prior to the UPC(s) being stopped...\n");
  xRC = PAPI_stop(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_stop...\n");
  else {
    printf("FAILURE:  PAPI_stop failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Perform a read of the counters after performing arithmetic, UPC is stopped...  Values should be the same as right after the prior PAPI_Stop()...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read...\n");
  else
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);
  Print_Counters(xEventSet);

  printf("\n==>  Zero local counters.  Perform a PAPI_accum, UPC is stopped...  Native values should be zero, and the local PAPI counters the same as the previous read...\n");
  Zero_Local_Counters(PAPI_Counters);
  xRC = PAPI_accum(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_accum...\n");
  }
  else {
    printf("FAILURE:  PAPI_accum failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Perform a PAPI_read, UPC is stopped...  All values should be zero...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_read...\n");
  }
  else {
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Perform a reset after performing arithmetic, UPC is stopped...  All values should be zero...\n");
  xRC = PAPI_reset(xEventSet);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_reset...\n");
  }
  else {
    printf("FAILURE:  PAPI_reset failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Perform another read of the counters after resetting the counters, UPC is stopped...  All values should be zero...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read...\n");
  else
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);
  Print_Counters(xEventSet);

  printf("\n==>  Perform another PAPI_accum after resetting the counters, UPC is stopped...  All values should be zero...\n");
  Zero_Local_Counters(PAPI_Counters);
  xRC = PAPI_accum(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_accum...\n");
  }
  else {
    printf("FAILURE:  PAPI_accum failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Perform another PAPI_read after accumulating and resetting the UPC, UPC is stopped...  All values should be zero...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_read...\n");
  }
  else {
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Start the UPC again...\n");
  xRC = PAPI_start(xEventSet);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_start...\n");
  else {
    printf("FAILURE:  PAPI_start failed, xRC=%d...\n", xRC);
    return;
  }

  FPUArith();

  printf("\n==>  Get the state of the event set...\n");
  xRC = PAPI_state(xEventSet, &xState);
  if (xRC == PAPI_OK) {
    if (xState == PAPI_RUNNING) {
      printf("SUCCESS:  PAPI_state is RUNNING...\n");
    }
    else {
      printf("FAILURE:  PAPI_state failed, incorrect state, xState=%d...\n", xState);
    }
  }
  else {
    printf("FAILURE:  PAPI_state failed, xRC=%d...\n", xRC);
    return;
  }

  printf("\n==>  Perform a read of the counters, UPC is running...  The individual native counter values will be greater than the PAPI counters because the PAPI counters are read prior to the reads for the individual counter values...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read...\n");
  else
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);
  Print_Counters(xEventSet);

  FPUArith();

  printf("\n==>  Perform another read of the counters, UPC is running...  Values should be increasing...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read...\n");
  else
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);
  Print_Counters(xEventSet);

  FPUArith();

  printf("\n==>  Perform another read of the counters, UPC is running...  Values should continue increasing...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read...\n");
  else
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);
  Print_Counters(xEventSet);

  printf("\n==>  Perform a reset after performing arithmetic, UPC is still running...  Native counter values should be less than prior read, but PAPI counter values should be identical to the prior read (local buffer was not changed)...\n");
  xRC = PAPI_reset(xEventSet);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_reset...\n");
  }
  else {
    printf("FAILURE:  PAPI_reset failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Zero local counters.  Perform a PAPI_accum, UPC is still running...\n");
  Zero_Local_Counters(PAPI_Counters);
  xRC = PAPI_accum(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_accum...\n");
  }
  else {
    printf("FAILURE:  PAPI_accum failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  FPUArith();

  printf("\n==>  Accumulate local counters.  Perform a PAPI_accum, UPC is still running...  PAPI counters should show an increase from prior accumulate...\n");
  xRC = PAPI_accum(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_accum...\n");
  }
  else {
    printf("FAILURE:  PAPI_accum failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  FPUArith();

  printf("\n==>  Accumulate local counters.  Perform another PAPI_accum, UPC is still running...  PAPI counters should show an increase from prior accumulate...\n");
  xRC = PAPI_accum(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_accum...\n");
  }
  else {
    printf("FAILURE:  PAPI_accum failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Zero local counters.  Perform a PAPI_accum, UPC is still running...  PAPI counters should be less than the prior accumulate...\n");
  Zero_Local_Counters(PAPI_Counters);
  xRC = PAPI_accum(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_accum...\n");
  }
  else {
    printf("FAILURE:  PAPI_accum failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Perform a PAPI_read, UPC is still running...  Native counters and PAPI counters should have both increased from prior accumulate...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_read...\n");
  }
  else {
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Perform a PAPI_write (not supported when UPC is running)...\n");
  xRC = PAPI_write(xEventSet, PAPI_Counters);
  if (xRC == PAPI_ECMP) {
    printf("SUCCESS:  PAPI_write, not allowed...\n");
  }
  else {
    printf("FAILURE:  PAPI_write failed, xRC=%d...\n", xRC);
    return;
  }

  printf("\n==>  Stop the UPC...  The individual native counter values will be greater than the PAPI counters because the PAPI counters are read prior to the UPC(s) being stopped...\n");
  xRC = PAPI_stop(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_stop...\n");
  else {
    printf("FAILURE:  PAPI_stop failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Perform a PAPI_read with the UPC stopped...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read...\n");
  else
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);

  printf("\n==>  Should be same 4 counters below, with the same native and PAPI counters as after the PAPI_stop...\n");
  Print_Counters(xEventSet);

  printf("\n==>  Perform a PAPI_accum with the UPC stopped...  Native counters sould be zeroed, with the PAPI counters unchanged from prior read (with the UPC already stopped, the accumulate does not add any counter values to the local buffer)...\n");
  Zero_Local_Counters(PAPI_Counters);
  xRC = PAPI_accum(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_accum...\n");
  }
  else {
    printf("FAILURE:  PAPI_accum failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Perform a PAPI_read with the UPC stopped...  Native and PAPI counters are zero...\n");
  xRC = PAPI_read(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read...\n");
  else
    printf("FAILURE:  PAPI_read failed, xRC=%d...\n", xRC);
  Print_Counters(xEventSet);

  printf("\n==>  Perform a reset, UPC is stopped...  Native and PAPI counters are zero...\n");
  xRC = PAPI_reset(xEventSet);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_reset...\n");
  }
  else {
    printf("FAILURE:  PAPI_reset failed, xRC=%d...\n", xRC);
    return;
  }
  Print_Counters(xEventSet);

  printf("\n==>  Perform a PAPI_write, but only to local memory...\n");
  xRC = PAPI_write(xEventSet, PAPI_Counters);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_write, but only to local memory...\n");
  }
  else {
    printf("FAILURE:  PAPI_write failed, xRC=%d...\n", xRC);
    return;
  }

  printf("\n==>  Get the state of the event set...\n");
  xRC = PAPI_state(xEventSet, &xState);
  if (xRC == PAPI_OK) {
    if (xState == PAPI_STOPPED) {
      printf("SUCCESS:  PAPI_state is STOPPED...\n");
    }
    else {
      printf("FAILURE:  PAPI_state failed, incorrect state, xState=%d...\n", xState);
    }
  }
  else {
    printf("FAILURE:  PAPI_state failed, xRC=%d...\n", xRC);
    return;
  }

  printf("\n==>  Get the multiplex status of the eventset...\n");
  xRC = PAPI_get_multiplex(xEventSet);
  if (xRC == PAPI_OK) {
    printf("SUCCESS:  PAPI_get_multiplex (NOTE:  The rest of the multiplex path is untested)...\n");
  }
  else {
    printf("FAILURE:  PAPI_get_multiplex failed, xRC=%d...\n", xRC);
    return;
  }

  printf("\n==>  Remove the events, and clean up the event set...\n");
  xRC = PAPI_remove_event(xEventSet, PNE_BGP_PU1_IPIPE_INSTRUCTIONS);
  if (xRC == PAPI_EINVAL)
    printf("SUCCESS:  PAPI_remove_event could not find PNE_BGP_PU1_IPIPE_INSTRUCTIONS...\n");
  else
    printf("FAILURE:  PAPI_remove_event PNE_BGP_PU1_IPIPE_INSTRUCTIONS failed, xRC=%d...\n", xRC);

  xRC = PAPI_remove_event(xEventSet, PAPI_L3_LDM);
  if (xRC == PAPI_EINVAL)
    printf("SUCCESS:  PAPI_remove_event could not find PAPI_L3_LDM...\n");
  else
    printf("FAILURE:  PAPI_remove_event PAPI_L3_LDM failed, xRC=%d...\n", xRC);

  xRC = PAPI_remove_event(xEventSet, PAPI_L1_TCM);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_remove_event PAPI_L1_TCM...\n");
  else
    printf("FAILURE:  PAPI_remove_event PAPI_L1_TCM failed, xRC=%d...\n", xRC);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 3)
    printf("SUCCESS:  PAPI_num_events returned 3...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_remove_event(xEventSet, PAPI_L1_ICM);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_remove_event PAPI_L1_ICM...\n");
  else
    printf("FAILURE:  PAPI_remove_event PAPI_L1_ICM failed, xRC=%d...\n", xRC);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 2)
    printf("SUCCESS:  PAPI_num_events returned 2...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_remove_event(xEventSet, PNE_BGP_PU3_L2_MEMORY_WRITES);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_remove_event PNE_BGP_PU3_L2_MEMORY_WRITES...\n");
  else
    printf("FAILURE:  PAPI_remove_event PNE_BGP_PU3_L2_MEMORY_WRITES failed, xRC=%d...\n", xRC);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 1)
    printf("SUCCESS:  PAPI_num_events returned 1...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_remove_event(xEventSet, PAPI_L1_DCM);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_remove_event PAPI_L1_DCM...\n");
  else
    printf("FAILURE:  PAPI_remove_event PAPI_L1_DCM failed, xRC=%d...\n", xRC);

  xRC = PAPI_num_events(xEventSet);
  if (xRC == 0)
    printf("SUCCESS:  PAPI_num_events returned 0...\n");
  else
    printf("FAILURE:  PAPI_num_events failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_cleanup_eventset(xEventSet);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_cleanup_eventset...\n");
  else
    printf("FAILURE:  PAPI_cleanup_eventset failed, xRC=%d...\n", xRC);

  xRC = PAPI_destroy_eventset(&xEventSet);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_destroy_eventset...\n");
  else
    printf("FAILURE:  PAPI_destroy_eventset failed, xRC=%d...\n", xRC);

  printf("==>  Do_Low_Level_Tests():  End of the main body...\n");

  return;
}

/*
 * Do_High_Level_Tests
 */

void Do_High_Level_Tests(void) {
  uint xEventId, xEventCode;
  int xRC, xNumEvents;

  printf("==>  Do_High_Level_Tests():  Beginning of the main body...\n");

  xRC = PAPI_num_counters();
  if (xRC == 256)
    printf("SUCCESS:  PAPI_num_counters returned 256 hardware counters...\n");
  else
    printf("FAILURE:  PAPI_num_counters failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_num_components();
  if (xRC == 1)
    printf("SUCCESS:  PAPI_num_components returned 256 hardware counters...\n");
  else
    printf("FAILURE:  PAPI_num_components failed, returned xRC=%d...\n", xRC);

  xEventId = 0;
  while (xEventId < MaxPresetEventId) {
    xNumEvents = 0;
    while (xEventId <= MaxPresetEventId && xNumEvents < NumEventsPerSet) {
      xEventCode = xEventId | 0x80000000;
      xRC = PAPI_query_event(xEventCode);
      if (xRC == PAPI_OK) {
        switch(xEventCode) {
          case 0x80000003:
          case 0x80000004:
          case 0x80000005:
          case 0x80000007:
          case 0x80000008:
          case 0x8000000A:
          case 0x8000000B:
          case 0x8000000C:
          case 0x8000000D:
          case 0x8000000F:
          case 0x80000010:
          case 0x80000011:
          case 0x80000012:
          case 0x80000013:
          case 0x80000014:
          case 0x80000015:
          case 0x80000016:
          case 0x80000017:
          case 0x80000018:
          case 0x80000019:
          case 0x8000001A:
          case 0x8000001B:
          case 0x8000001D:
          case 0x8000001E:
          case 0x8000001F:
          case 0x80000020:
          case 0x80000021:
          case 0x80000022:
          case 0x80000023:
          case 0x80000024:
          case 0x80000025:
          case 0x80000026:
          case 0x80000027:
          case 0x80000028:
          case 0x80000029:
          case 0x8000002A:
          case 0x8000002B:
          case 0x8000002C:
          case 0x8000002D:
          case 0x8000002E:
          case 0x8000002F:
          case 0x80000031:
          case 0x80000032:
          case 0x80000033:
          case 0x80000037:
          case 0x80000038:
          case 0x80000039:
          case 0x8000003A:
          case 0x8000003D:
          case 0x80000042:
          case 0x80000045:
          case 0x80000046:
          case 0x80000048:
          case 0x8000004A:
          case 0x8000004B:
          case 0x8000004D:
          case 0x8000004E:
          case 0x80000050:
          case 0x80000051:
          case 0x80000053:
          case 0x80000054:
          case 0x80000056:
          case 0x80000057:
          case 0x80000059:
          case 0x8000005c:
          case 0x8000005f:
          case 0x80000061:
          case 0x80000062:
          case 0x80000063:
          case 0x80000064:
          case 0x80000065:
            printf("FAILURE:  Do_High_Level_Tests, preset event code %#8.8x added to list of events to be started, but should not be allowed...\n", xEventCode);
            break;
          default:
            printf("SUCCESS:  Do_High_Level_Tests, preset event code %#8.8x added to list of events to be started...\n", xEventCode);
        }
        PAPI_Events[xNumEvents] = xEventCode;
        xNumEvents++;
      }
      else {
        switch(xEventCode) {
          case 0x80000003:
          case 0x80000004:
          case 0x80000005:
          case 0x80000007:
          case 0x80000008:
          case 0x8000000A:
          case 0x8000000B:
          case 0x8000000C:
          case 0x8000000D:
          case 0x8000000F:
          case 0x80000010:
          case 0x80000011:
          case 0x80000012:
          case 0x80000013:
          case 0x80000014:
          case 0x80000015:
          case 0x80000016:
          case 0x80000017:
          case 0x80000018:
          case 0x80000019:
          case 0x8000001A:
          case 0x8000001B:
          case 0x8000001D:
          case 0x8000001E:
          case 0x8000001F:
          case 0x80000020:
          case 0x80000021:
          case 0x80000022:
          case 0x80000023:
          case 0x80000024:
          case 0x80000025:
          case 0x80000026:
          case 0x80000027:
          case 0x80000028:
          case 0x80000029:
          case 0x8000002A:
          case 0x8000002B:
          case 0x8000002C:
          case 0x8000002D:
          case 0x8000002E:
          case 0x8000002F:
          case 0x80000031:
          case 0x80000032:
          case 0x80000033:
          case 0x80000037:
          case 0x80000038:
          case 0x80000039:
          case 0x8000003A:
          case 0x8000003D:
          case 0x80000042:
          case 0x80000045:
          case 0x80000046:
          case 0x80000048:
          case 0x8000004A:
          case 0x8000004B:
          case 0x8000004D:
          case 0x8000004E:
          case 0x80000050:
          case 0x80000051:
          case 0x80000053:
          case 0x80000054:
          case 0x80000056:
          case 0x80000057:
          case 0x80000059:
          case 0x8000005c:
          case 0x8000005f:
          case 0x80000061:
          case 0x80000062:
          case 0x80000063:
          case 0x80000064:
          case 0x80000065:
            printf("SUCCESS:  Do_High_Level_Tests, preset event code %#8.8x cannot be added to list of events to be started, xRC = %d...\n", xEventCode, xRC);
            break;
          default:
            printf("FAILURE:  Do_High_Level_Tests, preset event code %#8.8x cannot be added to list of events to be started, xRC = %d...\n", xEventCode, xRC);
        }
      }
      xEventId++;
    }
    if (xNumEvents)
      Run_Cycle(xNumEvents);
  }

  xEventId = 0;
  while (xEventId < MaxNativeEventId) {
    xNumEvents = 0;
    while (xEventId <= MaxNativeEventId && xNumEvents < NumEventsPerSet) {
      xEventCode = xEventId | 0x40000000;
      xRC = PAPI_query_event(xEventCode);
      if (xRC == PAPI_OK) {
        switch(xEventCode) {
          case 0x4000005C:
          case 0x4000005D:
          case 0x4000005E:
          case 0x4000005F:
          case 0x40000060:
          case 0x40000061:
          case 0x40000062:
          case 0x40000063:
          case 0x40000064:
          case 0x4000007C:
          case 0x4000007D:
          case 0x4000007E:
          case 0x4000007F:
          case 0x40000080:
          case 0x40000081:
          case 0x40000082:
          case 0x40000083:
          case 0x40000084:
          case 0x400000D8:
          case 0x400000D9:
          case 0x400000DA:
          case 0x400000DB:
          case 0x400000DC:
          case 0x400000DD:
          case 0x400000FD:
          case 0x400000FE:
          case 0x40000198:
          case 0x40000199:
          case 0x4000019A:
          case 0x4000019B:
          case 0x4000019C:
          case 0x4000019D:
          case 0x4000019E:
          case 0x4000019F:
          case 0x400001A0:
          case 0x400001B8:
          case 0x400001B9:
          case 0x400001BA:
          case 0x400001BB:
          case 0x400001BC:
          case 0x400001BD:
          case 0x400001BE:
          case 0x400001BF:
          case 0x400001C0:
          case 0x400001D2:
          case 0x400001D3:
          case 0x400001D4:
          case 0x400001D5:
          case 0x400001D6:
          case 0x400001D7:
          case 0x400001E6:
          case 0x400001E7:
          case 0x400001E8:
          case 0x400001E9:
          case 0x400001EA:
          case 0x400001EB:
          case 0x400001FE:
            printf("FAILURE:  Do_High_Level_Tests, native event code %#8.8x added to list of events to be started, but should not be allowed...\n", xEventCode);
            break;
          default:
            printf("SUCCESS:  Do_High_Level_Tests, native event code %#8.8x added to list of events to be started...\n", xEventCode);
        }
        PAPI_Events[xNumEvents] = xEventCode;
        xNumEvents++;
      }
      else {
        switch(xEventCode) {
          case 0x4000005C:
          case 0x4000005D:
          case 0x4000005E:
          case 0x4000005F:
          case 0x40000060:
          case 0x40000061:
          case 0x40000062:
          case 0x40000063:
          case 0x40000064:
          case 0x4000007C:
          case 0x4000007D:
          case 0x4000007E:
          case 0x4000007F:
          case 0x40000080:
          case 0x40000081:
          case 0x40000082:
          case 0x40000083:
          case 0x40000084:
          case 0x400000D8:
          case 0x400000D9:
          case 0x400000DA:
          case 0x400000DB:
          case 0x400000DC:
          case 0x400000DD:
          case 0x400000FD:
          case 0x400000FE:
          case 0x40000198:
          case 0x40000199:
          case 0x4000019A:
          case 0x4000019B:
          case 0x4000019C:
          case 0x4000019D:
          case 0x4000019E:
          case 0x4000019F:
          case 0x400001A0:
          case 0x400001B8:
          case 0x400001B9:
          case 0x400001BA:
          case 0x400001BB:
          case 0x400001BC:
          case 0x400001BD:
          case 0x400001BE:
          case 0x400001BF:
          case 0x400001C0:
          case 0x400001D2:
          case 0x400001D3:
          case 0x400001D4:
          case 0x400001D5:
          case 0x400001D6:
          case 0x400001D7:
          case 0x400001E6:
          case 0x400001E7:
          case 0x400001E8:
          case 0x400001E9:
          case 0x400001EA:
          case 0x400001EB:
          case 0x400001FE:
            printf("SUCCESS:  Do_High_Level_Tests, native event code %#8.8x cannot be added to list of events to be started, xRC = %d...\n", xEventCode, xRC);
            break;
          default:
            printf("FAILURE:  Do_High_Level_Tests, native event code %#8.8x cannot be added to list of events to be started, xRC = %d...\n", xEventCode, xRC);
        }
      }
      xEventId++;
    }
    if (xNumEvents)
      Run_Cycle(xNumEvents);
  }

  float xRtime, xPtime, xMflips, xMflops, xIpc;
  long long xFlpins, xFlpops, xIns;
  long long values[3] = {PAPI_FP_INS, PAPI_FP_OPS, PAPI_TOT_CYC};

  xRC = PAPI_flips(&xRtime, &xPtime, &xFlpins, &xMflips);

  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_flips started.\n");
  else
    printf("FAILURE:  PAPI_flips failed, returned xRC=%d...\n", xRC);

  FPUArith();

  xRC = PAPI_flips(&xRtime, &xPtime, &xFlpins, &xMflips);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_flips Rtime=%e Ptime=%e, Flpins=%lld, Mflips=%e\n", xRtime, xPtime, xFlpins, xMflips);
  else
    printf("FAILURE:  PAPI_flips failed, returned xRC=%d...\n", xRC);

  FPUArith();
  FPUArith();

  xRC = PAPI_flips(&xRtime, &xPtime, &xFlpins, &xMflips);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_flips Rtime=%e Ptime=%e, Flpins=%lld, Mflips=%e\n", xRtime, xPtime, xFlpins, xMflips);
  else
    printf("FAILURE:  PAPI_flips failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_stop_counters(values, 3);
  if (xRC ==  PAPI_OK)
    printf("SUCCESS:  PAPI_stop_counters stopped counters.\n");
  else
    printf("FAILURE:  PAPI_stop_counters failed, returned xRC=%d...\n", xRC);


  xRC = PAPI_flops(&xRtime, &xPtime, &xFlpops, &xMflops);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_flops started.\n");
  else
    printf("FAILURE:  PAPI_flops failed, returned xRC=%d...\n", xRC);

  FPUArith();
 
  xRC = PAPI_flops(&xRtime, &xPtime, &xFlpops, &xMflops);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_flops Rtime=%e Ptime=%e Flpops=%lld Mflops=%e\n", xRtime, xPtime, xFlpops, xMflops);
  else
    printf("FAILURE:  PAPI_flops failed, returned xRC=%d...\n", xRC);

  FPUArith();
  FPUArith();

  xRC = PAPI_flops(&xRtime, &xPtime, &xFlpops, &xMflops);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_flops Rtime=%e Ptime=%e Flpops=%lld Mflops=%e\n", xRtime, xPtime, xFlpops, xMflops);
  else
    printf("FAILURE:  PAPI_flops failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_stop_counters(values, 3);
  if (xRC ==  PAPI_OK)
    printf("SUCCESS:  PAPI_stop_counters stopped counters.\n");
  else
    printf("FAILURE:  PAPI_stop_counters failed, returned xRC=%d...\n", xRC);

  xRC = PAPI_ipc(&xRtime, &xPtime, &xIns, &xIpc);
  if (xRC == PAPI_ENOEVNT)
    printf("SUCCESS:  PAPI_ipc, no event found...\n");
  else
    printf("FAILURE:  PAPI_ipc failed, returned xRC=%d...\n", xRC);

  printf("==>  Do_High_Level_Tests():  End of the main body...\n");

  return;
}


/*
 * Do_Multiplex_Tests
 */

void Do_Multiplex_Tests(void) {
  int xRC;

  printf("==>  Do_Multiplex_Tests():  Beginning of the main body...\n");

  xRC = PAPI_multiplex_init();
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_multiplex_init...\n");
  else
    printf("FAILURE:  PAPI_multiplex_init failed, returned xRC=%d...\n", xRC);

  printf("==>  Do_Multiplex_Tests():  End of the main body...\n");

  return;
}


void Run_Cycle(const int pNumEvents) {
  int xRC;

//  BGP_UPC_Zero_Counter_Values();
  Zero_Local_Counters(PAPI_Counters);
  xRC = PAPI_start_counters(PAPI_Events, pNumEvents);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_start_counters...\n");
  else
    printf("FAILURE:  PAPI_start_counters failed, returned xRC=%d...\n", xRC);

  Print_Native_Counters();
  Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, pNumEvents);
  FPUArith();
  Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, pNumEvents);
  Print_PAPI_Counters_From_List(PAPI_Events, pNumEvents, PAPI_Counters);

  FPUArith();

  xRC = PAPI_read_counters(PAPI_Counters, pNumEvents);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read_counters...\n");
  else
    printf("FAILURE:  PAPI_read_counters failed, returned xRC=%d...\n", xRC);

  Print_Native_Counters();
  Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, pNumEvents);
  FPUArith();
  Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, pNumEvents);
  Print_PAPI_Counters_From_List(PAPI_Events, pNumEvents, PAPI_Counters);
  
  FPUArith();

  Zero_Local_Counters(PAPI_Counters);
  xRC = PAPI_accum_counters(PAPI_Counters, pNumEvents);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_accum_counters...\n");
  else
    printf("FAILURE:  PAPI_accum_counters failed, returned xRC=%d...\n", xRC);

  Print_Native_Counters();
  Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, pNumEvents);
  FPUArith();
  Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, pNumEvents);
  Print_PAPI_Counters_From_List(PAPI_Events, pNumEvents, PAPI_Counters);

  FPUArith();

  xRC = PAPI_read_counters(PAPI_Counters, pNumEvents);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_read_counters...\n");
  else
    printf("FAILURE:  PAPI_read_counters failed, returned xRC=%d...\n", xRC);

  Print_Native_Counters();
  Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, pNumEvents);
  FPUArith();
  Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, pNumEvents);
  Print_PAPI_Counters_From_List(PAPI_Events, pNumEvents, PAPI_Counters);

  FPUArith();

  xRC = PAPI_stop_counters(PAPI_Counters, pNumEvents);
  if (xRC == PAPI_OK)
    printf("SUCCESS:  PAPI_stop_counters...\n");
  else
    printf("FAILURE:  PAPI_stop_counters failed, returned xRC=%d...\n", xRC);

  Print_Native_Counters();
  Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, pNumEvents);
  FPUArith();
  Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, pNumEvents);
  Print_PAPI_Counters_From_List(PAPI_Events, pNumEvents, PAPI_Counters);

  FPUArith();

  return;
}


/*
 * Zero_Local_Counters
 */

void Zero_Local_Counters(long long* pCounters) {
  int i;
  for (i=0; i<255; i++)
  	pCounters[i] = 0;

  return;
}


/*
 * FPU Arithmetic...
 */
void FPUArith(void) {
  int i;

  printf("\n==>  Start:  Performing arithmetic...\n");
  register unsigned int zero = 0;
  register double *x_p = &x[0];

  for ( i = 0; i < 32; i++ )
    x[i] = 1.0;

  // Single Hummer Instructions:

  #if 1

  asm volatile ("fabs       1,2");
  asm volatile ("fmr        1,2");
  asm volatile ("fnabs      1,2");
  asm volatile ("fneg       1,2");

  asm volatile ("fadd       1,2,3");
  asm volatile ("fadds      1,2,3");
  asm volatile ("fdiv       1,2,3");
  asm volatile ("fdivs      1,2,3");
  asm volatile ("fmul       1,2,3");
  asm volatile ("fmuls      1,2,3");
  asm volatile ("fres       1,2");
  asm volatile ("frsqrte    1,2");
  //asm volatile ("fsqrt      1,2");          // gives exception
  //asm volatile ("fsqrts     1,2");          // gives exception
  asm volatile ("fsub       1,2,3");
  asm volatile ("fsubs      1,2,3");

  asm volatile ("fmadd      3,4,5,6");
  asm volatile ("fmadds     3,4,5,6");
  asm volatile ("fmsub      3,4,5,6");
  asm volatile ("fmsubs     3,4,5,6");
  asm volatile ("fnmadd     3,4,5,6");
  asm volatile ("fnmadds    3,4,5,6");
  asm volatile ("fnmsub     3,4,5,6");
  asm volatile ("fnmsubs    3,4,5,6");

  //asm volatile ("fcfid      5,6");          // invalid instruction
  //asm volatile ("fctid      5,6");          // invalid instruction
  //asm volatile ("fctidz     5,6");          // invalid instruction
  asm volatile ("fctiw      5,6");
  asm volatile ("fctiwz     5,6");
  asm volatile ("frsp       5,6");

  asm volatile ("fcmpo   0,1,2");
  asm volatile ("fcmpu   0,1,2");
  asm volatile ("fsel    0,1,2,3");

  #endif

  #if 1

  asm volatile("fpadd             9,10,11");
  asm volatile("fpsub             9,10,11");

  #endif


  #if 1

  asm volatile("fpmul            23,24,25");
  asm volatile("fxmul            26, 27, 28");
  asm volatile("fxpmul           28, 29, 30");
  asm volatile("fxsmul            2, 3, 4");
  #endif

  #if 1

  asm volatile("fpmadd           10,11,12,13");
  asm volatile("fpmsub           18, 19, 20, 21");
  asm volatile("fpnmadd          26, 27, 28, 29");
  asm volatile("fpnmsub          16,17,18,19");

  asm volatile("fxmadd           10,11,12,13");
  asm volatile("fxmsub           18, 19, 20, 21");
  asm volatile("fxnmadd          26, 27, 28, 29");
  asm volatile("fxnmsub          16,17,18,19");

  asm volatile("fxcpmadd           10,11,12,13");
  asm volatile("fxcpmsub           18, 19, 20, 21");
  asm volatile("fxcpnmadd          26, 27, 28, 29");
  asm volatile("fxcpnmsub          16,17,18,19");

  asm volatile("fxcsmadd           10,11,12,13");
  asm volatile("fxcsmsub           18, 19, 20, 21");
  asm volatile("fxcsnmadd          26, 27, 28, 29");
  asm volatile("fxcsnmsub          16,17,18,19");

  asm volatile("fxcpnpma           1,2,3,4");
  asm volatile("fxcsnpma           5,6,7,8");
  asm volatile("fxcpnsma           9,10,11,12");
  asm volatile("fxcsnsma           3,4,5,6");

  asm volatile("fxcxnpma           9,10,11,12");
  asm volatile("fxcxnsma           8,9,10,11");
  asm volatile("fxcxma             3,4,5,6");
  asm volatile("fxcxnms            8,9,10,11");

  #endif


  #if 1

  asm volatile("fpre               12, 13");
  asm volatile("fprsqrte           15, 16");
  asm volatile("fpsel              17, 18, 19, 20");
  asm volatile("fpctiw             1,2");
  asm volatile("fpctiwz            3,4");
  asm volatile("fprsp              5,6");
  asm volatile("fscmp              1,2,3");
  asm volatile("fpmr               1,2");
  asm volatile("fpneg              1,2");
  asm volatile("fpabs              1,2");
  asm volatile("fpnabs             1,2");
  asm volatile("fsmr               1,2");
  asm volatile("fsneg              1,2");
  asm volatile("fsabs              1,2");
  asm volatile("fsnabs             1,2");
  asm volatile("fxmr               1,2");
  asm volatile("fsmfp              1,2");
  asm volatile("fsmtp              1,2");

  #endif

  #if 1
  asm volatile("lfdx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfdux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfsx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfsux       16,%0,%1" : "+b"(x_p) : "b"(zero));

  asm volatile("lfsdx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfsdux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfssx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfssux       16,%0,%1" : "+b"(x_p) : "b"(zero));

  asm volatile("lfpsx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfpsux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfxsx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfxsux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  #endif

  #if 1
  asm volatile("lfpdx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfpdux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfxdx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("lfxdux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  #endif

  #if 1
  asm volatile("stfdx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfdux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfsx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfsux       16,%0,%1" : "+b"(x_p) : "b"(zero));

  asm volatile("stfsdx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfsdux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfssx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  //asm volatile("stfssux       16,%0,%1" : "+b"(x_p) : "b"(zero));

  asm volatile("stfpsx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfpsux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfxsx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfxsux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  #endif

  #if 1
  asm volatile("stfpdx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfpdux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfxdx       16,%0,%1" : "+b"(x_p) : "b"(zero));
  asm volatile("stfxdux       16,%0,%1" : "+b"(x_p) : "b"(zero));
  #endif
  printf("==>  End:    Performing arithmetic...\n");

  return;
}


/*
 * Print_Counters
 */
void Print_Counters(const int pEventSet) {
  printf("\n***** Start Print Counter Values *****\n");
//  Print_Native_Counters_via_Buffer((BGP_UPC_Read_Counters_Struct_t*)Native_Buffer);
//  Print_Native_Counters();
  Print_Native_Counters_for_PAPI_Counters(pEventSet);
  Print_PAPI_Counters(pEventSet, PAPI_Counters);
  printf("\n*****  End Print Counter Values  *****\n");

  return;
}


/*
 * Print_Native_Counters
 */

void Print_Native_Counters() {
  printf("\n***** Start Print of Native Counter Values *****\n");
  BGP_UPC_Print_Counter_Values(BGP_UPC_READ_EXCLUSIVE);
  printf("*****  End Print of Native Counter Values  *****\n");

  return;
}


/*
 * Print_Native_Counters_for_PAPI_Counters
 */

void Print_Native_Counters_for_PAPI_Counters(const int pEventSet) {
  printf("\n***** Start Print of Native Counter Values for PAPI Counters *****\n");
  int xNumEvents = PAPI_num_events(pEventSet);
  if (xNumEvents) {
    List_PAPI_Events(pEventSet, PAPI_Events, &xNumEvents);
    Print_Native_Counters_for_PAPI_Counters_From_List(PAPI_Events, xNumEvents);
  }
  else {
    printf("No events are present in the event set.\n");
  }
  printf("*****  End Print of Native Counter Values for PAPI Counters  *****\n");

  return;
}


/*
 * Print_Native_Counters_for_PAPI_Counters_From_List
 */
void Print_Native_Counters_for_PAPI_Counters_From_List(const int* pEvents, const int pNumEvents) {
  int i, j, xRC;
  char xName[256];
  BGP_UPC_Event_Id_t xNativeEventId;
  PAPI_event_info_t xEventInfo;

//  BGP_UPC_Print_Counter_Values(); // DLH
  for (i=0; i<pNumEvents; i++) {
    xRC = PAPI_event_code_to_name(PAPI_Events[i], xName);
    if (!xRC) {
      xRC = PAPI_get_event_info(PAPI_Events[i], &xEventInfo);
      if (xRC) {
        printf("FAILURE:  PAPI_get_event_info failed for %s, xRC=%d\n", xName, xRC);
        exit(1);
      }
      printf("\n     *** PAPI Counter Location %3.3d:  %#8.8x %s\n", i, PAPI_Events[i], xName);
      if (PAPI_Events[i] & 0x80000000) {
        // Preset event
        for (j=0; j<xEventInfo.count; j++) {
          xNativeEventId = (BGP_UPC_Event_Id_t)(xEventInfo.code[j]&0xBFFFFFFF);
//          printf("Preset:  j=%d, xEventInfo.code[j]=%#8.8x, xNativeEventId=%#8.8x\n", j, xEventInfo.code[j], xNativeEventId);
          BGP_UPC_Print_Counter_Value(xNativeEventId, BGP_UPC_READ_EXCLUSIVE);
        }
      }
      else {
        // Native event
        xNativeEventId = (BGP_UPC_Event_Id_t)(PAPI_Events[i]&0xBFFFFFFF);
//        printf("Native:  i=%d, PAPI_Events[i]=%#8.8x, xNativeEventId=%#8.8x\n", i, PAPI_Events[i], xNativeEventId);
        BGP_UPC_Print_Counter_Value(xNativeEventId, BGP_UPC_READ_EXCLUSIVE);
      }
    }
    else {
      printf("\n     *** PAPI Counter Location %3.3d:  Not mapped\n", i);
    }
  }
}


/*
 * Print_Native_Counters_via_Buffer
 */

void Print_Native_Counters_via_Buffer(const BGP_UPC_Read_Counters_Struct_t* pBuffer) {
  Read_Native_Counters(BGP_UPC_MAXIMUM_LENGTH_READ_COUNTERS_STRUCTURE);
  printf("\n***** Start Print of Native Counter Values *****\n");
  printf("Elapsed Running Time (native) = %lld\n", (*pBuffer).elapsed_time);
//  printf("Print_Native_Counters_via_Buffer:  Native_Buffer*=%p, pBuffer=%p, (*pBuffer).counter=%p\n", Native_Buffer, pBuffer, (*pBuffer).counter);
  Print_Counter_Values((long long*)(*pBuffer).counter, pBuffer->number_of_counters);
  printf("*****  End Print of Native Counter Values  *****\n");

  return;
}


/*
 * Print_PAPI_Counters
 */

void Print_PAPI_Counters(const int pEventSet, const long long* pCounters) {
  int i;
  char xName[256];
  printf("\n***** Start Print of PAPI Counter Values *****\n");
//  printf("Print_PAPI_Counters:  PAPI_Counters*=%p, pCounters*=%p\n", PAPI_Counters, pCounters);
  int pNumEvents = PAPI_num_events(pEventSet);
  printf("Number of Counters = %d\n", pNumEvents);
  if (pNumEvents) {
    printf("    Calculated Value Location Event Number Event Name\n");
    printf("-------------------- -------- ------------ --------------------------------------------\n");
    List_PAPI_Events(pEventSet, PAPI_Events, &pNumEvents);
    for (i=0; i<pNumEvents; i++) {
      if (PAPI_event_code_to_name(PAPI_Events[i], xName)) {
        printf("PAPI_event_code_to_name failed on event code %d\n", PAPI_Events[i]);
        exit(1);
      }
      printf("%20llu      %3d   %#8.8x %s\n", pCounters[i], i, PAPI_Events[i], xName);
    }
  }
  printf("*****  End Print of PAPI Counter Values  *****\n");

  return;
}


/*
 * Print_PAPI_Counters_From_List
 */

void Print_PAPI_Counters_From_List(const int* pEventList, const int pNumEvents, const long long* pCounters) {
  int i;
  char xName[256];
  printf("\n***** Start Print of PAPI Counter Values *****\n");
  printf("Number of Counters = %d\n", pNumEvents);
  if (pNumEvents) {
    printf("    Calculated Value Location Event Number Event Name\n");
    printf("-------------------- -------- ------------ --------------------------------------------\n");
    for (i=0; i<pNumEvents; i++) {
      if (PAPI_event_code_to_name(pEventList[i], xName)) {
        printf("PAPI_event_code_to_name failed on event code %d\n", pEventList[i]);
        exit(1);
      }
      printf("%20llu      %3d   %#8.8x %s\n", pCounters[i], i, pEventList[i], xName);
    }
  }
  printf("*****  End Print of PAPI Counter Values  *****\n");

  return;
}


/*
 * Print_Counter_Values
 */

void Print_Counter_Values(const long long* pCounters, const int pNumCounters) {
  int i=0, j, xEnd;
  long long xCounters[NUMBER_COUNTERS_PER_ROW];
  printf("Print_Counter_Values:  Native_Buffer*=%p, pCounters*=%p\n", Native_Buffer, pCounters);
  printf("Number of Counters = %d\n", pNumCounters);
  if (pNumCounters) {
    printf("                     +0        +1        +2        +3        +4        +5        +6        +7\n");
    printf("---------------------------------------------------------------------------------------------\n");
    xEnd = (((pNumCounters-1)/NUMBER_COUNTERS_PER_ROW)*NUMBER_COUNTERS_PER_ROW)+NUMBER_COUNTERS_PER_ROW;
    while (i < xEnd) {
      for (j=0; j<NUMBER_COUNTERS_PER_ROW; j++) {
        if (i+j < pNumCounters) {
          xCounters[j] = pCounters[i+j];
        }
        else
          xCounters[j] = -1;
      }
      printf("Ctrs %3.3d-%3.3d:  %8lld  %8lld  %8lld  %8lld  %8lld  %8lld  %8lld  %8lld\n",
               i, i+7, xCounters[0], xCounters[1], xCounters[2], xCounters[3], xCounters[4],
               xCounters[5], xCounters[6], xCounters[7]);
      i += NUMBER_COUNTERS_PER_ROW;
    }
  }

  return;
}
/*
 * Print_Node_info
 */

void Print_Node_Info(void) {
  Read_Native_Counters(BGP_UPC_MINIMUM_LENGTH_READ_COUNTERS_STRUCTURE);
  BGP_UPC_Read_Counters_Struct_t* xTemp;
  xTemp = (BGP_UPC_Read_Counters_Struct_t*)(void*)Native_Buffer;
  printf("***** Start Print of Node Information *****\n");
  printf("Rank = %d\n", xTemp->rank);
  printf("Core = %d\n", xTemp->core);
  printf("UPC Number = %d\n", xTemp->upc_number);
  printf("Number of Processes per UPC = %d\n", xTemp->number_processes_per_upc);
  printf("User Mode = %d\n", (int) xTemp->mode);
  printf("Location = %s\n", xTemp->location);
  printf("\n*****  End Print of Node Information *****\n\n");

  return;
}


/*
 * Read_Native_Counters
 */

void Read_Native_Counters(const int pLength) {

  int xRC = BGP_UPC_Read_Counter_Values(Native_Buffer, pLength, BGP_UPC_READ_EXCLUSIVE);
  if (xRC < 0) {
    printf("FAILURE:  BGP_UPC_Read_Counter_Values failed, xRC=%d...\n", xRC);
    exit(1);
  }

  return;
}

/*
 * Print_PAPI_Events
 */

void Print_PAPI_Events(const int pEventSet) {
  int i;
  char xName[256];
  int pNumEvents = PAPI_num_events(pEventSet);
  List_PAPI_Events(pEventSet, PAPI_Events, &pNumEvents);
  for (i=0; i<pNumEvents; i++) {
    if (!PAPI_event_code_to_name(PAPI_Events[i], xName))
      printf("PAPI Counter Location %3.3d:  %#8.8x %s\n", i, PAPI_Events[i], xName);
    else
      printf("PAPI Counter Location %3.3d:  Not mapped\n", i);
  }

  return;
}


/*
 * List_PAPI_Events
 */
void List_PAPI_Events(const int pEventSet, int* pEvents, int* pNumEvents) {
  int xRC = PAPI_list_events(pEventSet, pEvents, pNumEvents);
  if (xRC != PAPI_OK) {
    printf("FAILURE:  PAPI_list_events failed, returned xRC=%d...\n", xRC);
    exit(1);
  }

  return;
}


/*
 * DumpInHex
 */
void DumpInHex(const char* pBuffer, int pSize) {

  return;
}

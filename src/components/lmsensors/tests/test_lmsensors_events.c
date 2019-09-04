/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    test_lmsensors_events.c
 * 
 * @author  Rizwan Ashraf
 *          rizwan@icl.utk.edu
 *       
 * Test all events reported by the lmsensors component.
 *      
 * @brief
 *       The code automatically checks if the lmsensors component is enabled and 
 *       correspondingly adds all available PAPI events in the event set, one at a time.
 *       The event values are recorded in each case, and listed at the completion of the 
 *       test. The event values need to be checked manually for correctness. 
 *
 *       Usage: ./test_lmsensors_events
 */
 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* headers required by PAPI */
#include "papi.h"
#include "papi_test.h"

/* constants */
// No. of double floating point computations
#define NSIZE 10000
// No. of times computation should be repeated 
#define REPEAT 1
// The max no. of lmsensor events expected
#define MAX_LMSENSORS_EVENTS 150
// EVENT_* : No. of events out of all possible events as reported by
//           component_info which need to be added successfully to the
//           event set. PASS_THRESHOLD defines the PASS criteria. 
//           If PASS_THRESHOLD is not met, then this threshold is used 
//           to check if the test can be declared PASS WITH WARNING. 
//           Otherwise, the test is declared as FAILED.
#define EVENT_PASS_THRESHOLD 90
#define EVENT_WARN_THRESHOLD 50

int main (int argc, char **argv) {

   /* Set TESTS_QUIET variable */
   tests_quiet( argc, argv );
   
   /*************************  SETUP PAPI ENV *************************************
    *******************************************************************************/                              
   int retVal, r, code;
   int ComponentID, NumComponents, LMSENSORS_ID = -1;
   int EventSet = PAPI_NULL;
   int eventCount = 0;  // total events as reported by component info
   int eventNum = 0;    // number of events successfully tested

   /* error reporting */
   int addEventFailCount = 0, codeConvertFailCount = 0, eventInfoFailCount = 0;  
   int PAPIstartFailCount = 0, PAPIstopFailCount = 0;
   int failedEventCodes[MAX_LMSENSORS_EVENTS];
   int failedEventIndex = 0;

   /* Note: these are fixed length arrays */ 
   char eventNames[MAX_LMSENSORS_EVENTS][PAPI_MAX_STR_LEN];
   char description[MAX_LMSENSORS_EVENTS][PAPI_MAX_STR_LEN];
   long long values[REPEAT][MAX_LMSENSORS_EVENTS];

   const PAPI_component_info_t *cmpInfo = NULL;
   PAPI_event_info_t eventInfo;
   
   /* for timing the test */
   long long startTime, endTime;
   double elapsedTime;

   /* PAPI Initialization */
   retVal = PAPI_library_init( PAPI_VER_CURRENT );
   if ( retVal != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__,"PAPI_library_init failed. The test has been terminated.\n",retVal);
   }
   
   /* Get total number of components detected by PAPI */ 
   NumComponents = PAPI_num_components();

   /* Check if lmsensors component exists */
   for ( ComponentID = 0; ComponentID < NumComponents; ComponentID++ ) {
      
        if ( (cmpInfo = PAPI_get_component_info(ComponentID)) == NULL ) {
            fprintf(stderr, "WARNING: PAPI_get_component_info failed on one of the components.\n"
                            "\t The test will continue for now, but it will be skipped later on\n"
                            "\t if this error was for a component under test.\n");
            continue;
        }

        if (strcmp(cmpInfo->name, "lmsensors") != 0) {
            continue;
        }

        // if we are here, lmsensors component is found 
        if (!TESTS_QUIET) {
            printf("INFO: Component %d (%d) - %d events - %s\n",
                ComponentID, cmpInfo->CmpIdx,
                cmpInfo->num_native_events, cmpInfo->name);
        }

        if (cmpInfo->disabled) {
            test_skip(__FILE__,__LINE__,"lmsensors Component is disabled. The test has been terminated.\n", 0);
            break;
        }

        eventCount = cmpInfo->num_native_events;
        LMSENSORS_ID = ComponentID;
        break;
   }
 
   /* if we did not find any valid events, just skip the test. */
   if (eventCount==0) {
       fprintf(stderr, "FATAL: No events found for the lmsensors component, even though it is enabled.\n"
                       "       The test will be skipped.\n"); 
       test_skip(__FILE__,__LINE__,"No events found for the lmsensors component.\n", 0);
   }

   /*************************  SETUP COMPUTE ENV **************************************
    *******************************************************************************/
   
   int i, j, k;  // loop variables

   /* data array */
   double *X, *Y, *Out;
   
   X = (double *) malloc (sizeof(double) * NSIZE);
   Y = (double *) malloc (sizeof(double) * NSIZE);
   Out = (double *) malloc (sizeof(double) * NSIZE);
       
   // check if memory was successfully allocated.
   if ( (X == NULL) || (Y == NULL) || (Out == NULL) ) {
        test_fail(__FILE__,__LINE__,"Could not allocate memory during the test. This is fatal and the test has been terminated.\n", 0);
   }
   
   /* initial data arrays */    
   for ( i = 0; i < NSIZE; i++ ) {
        X[i] = i*0.25;
        Y[i] = i*0.75;
   }

   if (!TESTS_QUIET) 
       printf("INFO: Success in initializing data arrays.\n");

   /*************************  MAIN TEST CODE *************************************   
   *******************************************************************************/
   startTime = PAPI_get_real_nsec();

   /* create an eventSet */
   retVal = PAPI_create_eventset ( &EventSet );
   if (retVal != PAPI_OK) {
       // handle error cases for PAPI_create_eventset() 
       // Two outcomes are possible here:
       // 1. PAPI_EINVAL: invalid argument. This should not occur.
       // 2. PAPI_ENOMEM: insufficient memory. If this is the case, then we need to quit the test.
       fprintf(stderr, "FATAL: Could not create an eventSet due to: %s.\n"
                       "       Test will not proceed.\n", PAPI_strerror(retVal));
       test_fail(__FILE__, __LINE__, "PAPI_create_eventset failed. This is fatal and the test has been terminated.\n", retVal);
   } // end -- handle error cases for PAPI_create_eventset()

   /* find the code for first event in component */
   code = PAPI_NATIVE_MASK;
   r = PAPI_enum_cmp_event ( &code, PAPI_ENUM_FIRST, LMSENSORS_ID );
   
   /* add each event individually in the eventSet and measure event values. */
   /* for each event, work is repeated. */
   while ( r == PAPI_OK ) {

      // attempt to add event to event set 
      retVal = PAPI_add_event (EventSet, code);
      if (retVal != PAPI_OK ) {
          // handle error cases for PAPI_add_event()
          if (retVal == PAPI_ENOMEM) {  
              fprintf(stderr, "FATAL: Could not add an event to eventSet due to insufficient memory.\n"
                              "       Test will not proceed.\n");
              test_fail(__FILE__, __LINE__, "PAPI_add_event failed due to fatal error and the test has been terminated.\n", retVal);
          }

          if (retVal == PAPI_ENOEVST) {
              fprintf(stderr, "WARNING: Could not add an event to eventSet since eventSet does not exist.\n"
                              "\t Test will proceed attempting to create a new eventSet\n");
              EventSet = PAPI_NULL;  
              retVal = PAPI_create_eventset ( &EventSet );
              if (retVal != PAPI_OK)
                  test_fail(__FILE__, __LINE__, "PAPI_create_eventset failed while handling failure of PAPI_add_event." 
                                                " This is fatal and the test has been terminated.\n", retVal);
              continue;  
          }
          
          if (retVal == PAPI_EISRUN) {
              long long tempValue;
              fprintf(stderr, "WARNING: Could not add an event to eventSet since eventSet is already counting.\n"
                              "\t Test will proceed attempting to stop counting and re-attempting to add current event.\n");
              retVal = PAPI_stop (EventSet, &tempValue);
              if (retVal != PAPI_OK) 
                  test_fail(__FILE__,__LINE__,"PAPI_stop failed while handling failure of PAPI_add_event." 
                                              " This is fatal and the test has been terminated.\n", retVal);  
              retVal = PAPI_cleanup_eventset( EventSet );
              if (retVal != PAPI_OK)
                  test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset failed while handling failure of PAPI_add_event."
                                              " This is fatal and the test has been terminated.\n", retVal);
              continue;              
          }
 
          // for all other errors, skip an event
          addEventFailCount++; // error reporting  
          failedEventCodes[failedEventIndex] = code;
          failedEventIndex++;        
          fprintf(stderr, "WARNING: Could not add an event to eventSet due to: %s.\n" 
                           "\t Test will proceed attempting to add other events.\n", PAPI_strerror(retVal));
         
          r = PAPI_enum_cmp_event (&code, PAPI_ENUM_EVENTS, LMSENSORS_ID);
          
          if (addEventFailCount >= eventCount) // if no event was added successfully
               break;  

          continue; 
      }  // end -- handle error cases for PAPI_add_event()
    
      /* get event name of added event */
      retVal = PAPI_event_code_to_name (code, eventNames[eventNum]);
      if (retVal != PAPI_OK ) {
          // handle error cases for PAPI_event_code_to_name().
          codeConvertFailCount++; // error reporting
          fprintf(stderr, "WARNING: PAPI_event_code_to_name failed due to: %s.\n"
                          "\t Test will proceed but an event name will not be available.\n", PAPI_strerror(retVal));
          strncpy(eventNames[eventNum], "ERROR:NOT_AVAILABLE", sizeof(eventNames[0])-1);
          eventNames[eventNum][sizeof(eventNames[0])-1] = '\0'; 
      } // end -- handle error cases for PAPI_event_code_to_name()
      
      /* get long description of added event */
      retVal = PAPI_get_event_info (code, &eventInfo);
      if (retVal != PAPI_OK ) {
          // handle error cases for PAPI_get_event_info()
          eventInfoFailCount++;  // error reporting
          fprintf(stderr, "WARNING: PAPI_get_event_info failed due to: %s.\n"
                          "\t Test will proceed but an event description will not be available.\n", PAPI_strerror(retVal));
          strncpy(description[eventNum], "ERROR:NOT_AVAILABLE", sizeof(description[0])-1);
          description[eventNum][sizeof(description[0])-1] = '\0';
      } else {
          strncpy(description[eventNum], eventInfo.long_descr, sizeof(description[0])-1);
          description[eventNum][sizeof(description[0])-1] = '\0';
      }
      
      /****************** PERFORM WORK (REPEATED) ************************************
       *******************************************************************************/
      for (i = 0; i < REPEAT; i++) { 

          /* start recording event value */
          retVal = PAPI_start (EventSet);
          if (retVal != PAPI_OK ) {
              // handle error cases for PAPI_start()
              // we need to skip the current event being counted for all errors,
              // in all cases, errors will be handled later on.

              PAPIstartFailCount++;  // error reporting
              failedEventCodes[failedEventIndex] = code;
              failedEventIndex++;
              fprintf(stderr, "WARNING: PAPI_start failed on Event Number %d (%s) due to: %s.\n"
                              "\t Test will proceed with other events if available.\n",
                              eventNum, eventNames[eventNum], PAPI_strerror(retVal));
              
              for (k = i; k < REPEAT; k++)  // fill invalid event values.
                   values[k][eventNum] = (unsigned long long) - 1; 
              
              break;  // try next event
          } // end -- handle error cases for PAPI_start()

	  if (!TESTS_QUIET)
              printf("INFO: Doing some computation on data arrays for event: %s.\n", eventNames[eventNum]);

          /* perform calculation. */
          /* Note: there is redundant computation here. */ 
          for (j = 0; j < NSIZE; j++)
               Out [j] = X [j] + Y [j];

          /* stop recording and collect event value */ 
          retVal = PAPI_stop (EventSet, &values[i][eventNum]);
          if (retVal != PAPI_OK ) {
              // handle error cases for PAPI_stop()
              // we need to skip the current event for all errors 
              // except one case, as below.
              PAPIstopFailCount++;  // error reporting
              if (retVal == PAPI_ENOTRUN) { 
                  fprintf(stderr, "WARNING: PAPI_stop failed on Event Number %d (%s) since eventSet is not running.\n"
                              "\t Test will attempt to restart counting on this eventSet.\n",
                              eventNum, eventNames[eventNum]);
                  if (PAPIstopFailCount < REPEAT) { 
                      i = i - 1; // re-attempt
                      continue;
                  }
              }
              // for all other errors, try next event.
              failedEventCodes[failedEventIndex] = code;
              failedEventIndex++;
              fprintf(stderr, "WARNING: PAPI_stop failed on Event Number %d (%s) due to: %s.\n"
                              "\t Test will proceed with other events if available.\n",
                              eventNum, eventNames[eventNum], PAPI_strerror(retVal));
              
              for (k = i; k < REPEAT; k++)   // fill invalid event values
                   values[k][eventNum] = (unsigned long long) - 1;

              break;  
          }  // end -- handle error cases for PAPI_stop()

      }   // end -- work loop
  
      /* Done, clean up eventSet for next iteration */
      retVal = PAPI_cleanup_eventset( EventSet );
      if (retVal != PAPI_OK) {
          // handle failure cases for PAPI_cleanup_eventset()
          if (retVal == PAPI_ENOEVST) {
              fprintf(stderr, "WARNING: Could not clean up eventSet since eventSet does not exist.\n"
                              "\t Test will proceed attempting to create a new eventSet\n");
              EventSet = PAPI_NULL;
              retVal = PAPI_create_eventset ( &EventSet );
              if (retVal != PAPI_OK)
                  test_fail(__FILE__, __LINE__, "PAPI_create_eventset failed while handling failure of PAPI_cleanup_eventset.\n"
                                                "This is fatal and the test has been terminated.\n", retVal);                                                    
          } else if (retVal == PAPI_EISRUN) {
              long long tempValue;
              fprintf(stderr, "WARNING: Could not clean up eventSet since eventSet is already counting.\n"
                              "\t Test will proceed attempting to stop counting and re-attempting to clean up.\n");
              retVal = PAPI_stop (EventSet, &tempValue);
              if (retVal != PAPI_OK)
                  test_fail(__FILE__,__LINE__,"PAPI_stop failed while handling failure of PAPI_cleanup_eventset."
                                              "This is fatal and the test has been terminated.\n", retVal);
              retVal = PAPI_cleanup_eventset( EventSet );
              if (retVal != PAPI_OK)
                  test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset failed once again while handling failure of PAPI_cleanup_eventset."
                                              "This is fatal and the test has been terminated.\n", retVal);
          } else {
              test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset failed:", retVal);
          }
      } // end -- handle failure cases for PAPI_cleanup_eventset()
  
      /* get next event */
      eventNum++;
      r = PAPI_enum_cmp_event (&code, PAPI_ENUM_EVENTS, LMSENSORS_ID);
           
   } // end -- event loop

   /* Done, destroy eventSet */
   retVal = PAPI_destroy_eventset( &EventSet );
   if (retVal != PAPI_OK) {
       // handle error cases for PAPI_destroy_eventset() 
       if (retVal == PAPI_ENOEVST || retVal == PAPI_EINVAL) {
           fprintf(stderr, "WARNING: Could not destroy eventSet since eventSet does not exist or has invalid value.\n"
                           "\t Test will proceed with other operations.\n");
       } else if (retVal == PAPI_EISRUN) {
           long long tempValue;
           fprintf(stderr, "WARNING: Could not destroy eventSet since eventSet is already counting.\n"
                           "\t Test will proceed attempting to stop counting and re-attempting to clean up.\n");
           retVal = PAPI_stop (EventSet, &tempValue);
           if (retVal != PAPI_OK)
               test_fail(__FILE__,__LINE__,"PAPI_stop failed while handling failure of PAPI_destroy_eventset."
                                            "This is fatal and the test has been terminated.\n", retVal);
           retVal = PAPI_cleanup_eventset( EventSet );
           if (retVal != PAPI_OK)
               test_fail(__FILE__,__LINE__,"PAPI_cleanup_eventset failed while handling failure of PAPI_destroy_eventset."
                                           "This is fatal and the test has been terminated.\n", retVal); 
           retVal = PAPI_destroy_eventset(&EventSet);
           if (retVal != PAPI_OK)
               test_fail(__FILE__,__LINE__,"PAPI_destroy_eventset failed once again while handling failure of PAPI_destroy_eventset."
                                           " This is fatal and the test has been terminated.\n", retVal);
       } else {
           fprintf(stderr, "WARNING: Could not destroy eventSet since there is an internal bug in PAPI.\n"
                           "\t Please report this to the developers. Test will proceed and operation may be unexpected.\n");
       }
   }  // end -- handle failure cases for PAPI_destroy_eventset() 

   /*************************** SUMMARIZE RESULTS ********************************
    ******************************************************************************/
   endTime = PAPI_get_real_nsec();
   elapsedTime = ((double) (endTime-startTime))/1.0e9;

   /* print results: event values and descriptions */
   if (!TESTS_QUIET) {
       int eventX;
       // print event values 
       printf("POST WORK EVENT VALUES (Event Name, Event Value)>>>\n");
       for (eventX = 0; eventX < eventNum; eventX++) {
           printf("\t%s --> \t\t", eventNames[eventX]);
           for (i = 0; i < REPEAT; i++) {
                if (i < REPEAT-1) 
                    printf("%.1f, ", (float) values[i][eventX]/1000.0);
                else
                    printf("%.1f.", (float) values[i][eventX]/1000.0);
           }
           printf("\n");
       }
         
       // print description of each event 
       printf("\n\nTHE DESCRIPTION OF EVENTS IS AS FOLLOWS>>>\n"); 
       for (eventX = 0; eventX < eventNum; eventX++) {
            printf("\t%s \t\t--> %s \n", eventNames[eventX], description[eventX]);
       }
       
   }
   
   /* test summary: 1) sanity check on floating point computation */
   int computeTestPass = 0, computeTestPassCount = 0;
   // check results of computation 
   for (i = 0; i < NSIZE; i++) {
        if ( fabs(Out[i] - (X[i] + Y[i])) < 0.00001 )
            computeTestPassCount++;
   }
   // summarize results of computation
   if (computeTestPassCount == NSIZE)
       computeTestPass = 1;

   // free memory
   free (X); free (Y); free (Out);
   
   /* test summary: 2) number of events added and counted successfully */
   int eventNumTestPass = 0;
   // test evaluation criteria
   if (((float) failedEventIndex / eventCount) <= (1.0 - (float) EVENT_PASS_THRESHOLD/100) )
       eventNumTestPass = 1;
   else if (((float) failedEventIndex / eventCount) <= (1.0 - (float) EVENT_WARN_THRESHOLD/100) ) 
       eventNumTestPass = -1; 
   else
       eventNumTestPass = 0;


   /* print test summary */
   if (!TESTS_QUIET) {

       printf("\n\n************************ TEST SUMMARY (EVENTS) ******************************\n"
              "No. of Events NOT tested successfully: %d (%.1f%%)\n"
              "Total No. of Events reported by component info: %d\n", 
              failedEventIndex, ((float) failedEventIndex/eventCount)*100.0, eventCount);
       
       if (failedEventIndex > 0) {
           printf("\tNames of Events NOT tested: ");
           char failedEventName[PAPI_MAX_STR_LEN];
           for (i = 0; i < failedEventIndex; i++) {
                retVal = PAPI_event_code_to_name (failedEventCodes[i], failedEventName);
                if (retVal != PAPI_OK) {
                    strncpy(failedEventName, "ERROR:NOT_AVAILABLE", sizeof(failedEventName)-1);
                    failedEventName[sizeof(failedEventName)-1] = '\0';
                }
                printf("%s ", failedEventName);
                if ((i > 0) && (i % 2 == 1)) printf("\n   \t\t\t\t");
           }
           printf("\n");

           printf("\tThe error counts for different PAPI routines are as follows:\n"
                  "\t\t\tNo. of PAPI add event errors (major) --> %d\n"
                  "\t\t\tNo. of PAPI code convert errors (minor) --> %d\n"
                  "\t\t\tNo. of PAPI event info errors (minor) --> %d\n" 
                  "\t\t\tNo. of PAPI start errors (major) --> %d\n"
                  "\t\t\tNo. of PAPI stop errors (major) --> %d\n", 
                  addEventFailCount, codeConvertFailCount, eventInfoFailCount, PAPIstartFailCount, PAPIstopFailCount);
       }
       printf("The PAPI event test has ");
       if (eventNumTestPass == 1) printf("PASSED\n");
       else if (eventNumTestPass == -1) printf("PASSED WITH WARNING\n");
       else printf("FAILED\n");
 
       // compute values
       printf("************************ TEST SUMMARY (COMPUTE VALUES) **********************\n");
       if (computeTestPassCount != NSIZE) {
           printf("No. of times sanity check FAILED on the floating point computation: %d (%.1f%%)\n"
                  "Total No. of floating point computations performed: %d \n",
                  NSIZE-computeTestPassCount, ((float) (NSIZE-computeTestPassCount)/NSIZE)*100.0, NSIZE); 
       } else {
           printf("Sanity check PASSED on all floating point computations.\n"
                  "Note: this may pass even if one event was tested successfully!\n");
       }
       printf("The overall test took %.3f secs.\n\n", elapsedTime);           
   } // end -- print summary of test results.

   /* determine success of overall test based on all tests */
   if (computeTestPass == 1 && eventNumTestPass == 1) {
       // all has to be good for the test to pass. 
       // note: test will generate a warning if tx/rx events are not available.
       test_pass( __FILE__ );
   } 
   else if (eventNumTestPass < 0) { 
       test_warn(__FILE__,__LINE__,"A warning was generated during any PAPI related tests or sanity check on computation failed", 0);
       test_pass(__FILE__);
   }
   else { 
       // fail, in case eventNumTest has failed, 
       // irrespective of the result of computeTest. 
       test_fail(__FILE__, __LINE__,"Any of PAPI event related tests have failed", 0);
   }

} // end main


/*
 * DADD API interface
 */

/*
 * Author:  Paul J. Drongowski
 * Address: DUDE
 *          Compaq Computer Corporation
 *          110 Spit Brook Road
 *          Nashua, NH
 * Date:    26 April 2002
 * Version: 1.2
 *
 * Copyright (c) 2002 Compaq Computer Corporation
 *
 */

/*
 * Function: dadd_terminate_cleanup
 * Purpose: Closes down server connection and terminates program.
 * Arguments:
 *   ignore:
 * Returns: Nothing; Modifies server as a side-effect.
 */

extern void dadd_terminate_cleanup(int ignore) ;

/*
 * Function: dadd_start_monitoring
 * Purpose: Set-up DADD monitoring for a particular PID
 * Arguments:
 *   pid: The process to be monitored
 * Returns: Pointer to the shared memory region, 0 on failure
 *
 * This function could be split out and put into pcdlib.c. It might be
 * necessary to return the server handle using a call by reference argument.
 */

extern unsigned char *dadd_start_monitoring(pid_t pid) ;

/*
 * Function: dadd_stop_monitoring
 * Purpose: Terminate DADD monitoring for a particular PID
 * Arguments:
 *   pid: The process being monitored
 *   region_address: Pointer to the shared memory region
 * Returns: -1 on failure, 0 on success
 *
 * This function could be split out and put into pcdlib.c. It might be
 * necessary to pass the server handle as an argument.
 */

extern int dadd_stop_monitoring(pid_t pid, unsigned char *region_address) ;

/* $Id$
 * command.c
 *
 * This test program illustrates how a process may use the perfctr
 * inheritance feature of the Linux x86 Performance-Monitoring Counters
 * interface to monitor the execution of an arbitrary command.
 *
 * Usage: ./command path-to-program [options-for-program]
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include "libperfctr.h"

int main(int argc, char **argv)
{
    struct perfctr_dev *dev;
    struct vperfctr *parent;
    struct vperfctr_state state;
    struct perfctr_control control;
    struct perfctr_info info;
    unsigned nrctrs; /* > 1 if we have PMCs in addition to a TSC */
    int pid;

    /*
     * Check that we have a command to run.
     */
    if( argc < 2 ) {
	fprintf(stderr, "no command to run?\n");
	return 1;
    }

    /*
     * Parent sets up the perfctrs.
     */
    if( (dev = perfctr_dev_open()) == NULL ) {
	perror("perfctr_dev_open");
	return 1;
    }
    if( perfctr_info(dev, &info) ) {
	perror("perfctr_info");
	return 1;
    }
    nrctrs = perfctr_cpu_nrctrs(&info);
    if( (parent = vperfctr_attach(dev)) == NULL ) {
	perror("vperfctr_attach");
	return 1;
    }
    memset(&control, 0, sizeof control);
    if( nrctrs > 1 )
	control.evntsel[0] = perfctr_evntsel_num_insns(&info);
    if( vperfctr_control(parent, &control) < 0 ) {
	perror("vperfctr_control");
	return 1;
    }
    perfctr_dev_close(dev);

    /*
     * Fork, and let child exec the command.
     */
    if( (pid = fork()) < 0 ) {
	perror("fork");
	return 1;
    } else if( pid == 0 ) {
	vperfctr_close(parent);
	execvp(argv[1], argv+1);
	perror(argv[1]);
	_exit(1);
    }

    /*
     * Parent waits for child's exit and reports final perfctr values.
     */
    waitpid(pid, NULL, 0); /* XXX: should check for non-successful exits */
    if( vperfctr_read_state(parent, &state) < 0 ) {
	perror("vperfctr_read_state");
	return 1;
    }
    printf("tsc\t0x%016llX\n", state.children.ctr[0]);
    if( nrctrs > 1 )
	printf("pmc[0]\t0x%016llX\n", state.children.ctr[1]);
    return 0;
}

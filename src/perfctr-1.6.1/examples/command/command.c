/* $Id$
 * command.c
 *
 * This test program illustrates how a process may use the
 * Linux x86 Performance-Monitoring Counters interface to
 * monitor the execution of an arbitrary command.
 *
 * Usage: ./command path-to-program [options-for-program]
 *
 * Copyright (C) 1999-2000  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include "libperfctr.h"

static unsigned nrctrs; /* > 1 if we have PMCs in addition to a TSC */

void init(void)
{
    nrctrs = perfctr_cpu_nrctrs();
}

int child(int pipe_from_parent, char **argv)
{
    char c;

    read(pipe_from_parent, &c, 1);	/* wait until parent says 'go' */
    close(pipe_from_parent);
    execvp(argv[0], argv);
    perror(argv[0]);
    _exit(1);
}

int parent(int pipe_to_child, int child_pid)
{
    struct vperfctr_state state;
    struct vperfctr *perfctr;

    memset(&state.control, 0, sizeof state.control);
    if( nrctrs > 1 )
	state.control.evntsel[0] = perfctr_evntsel_num_insns();

    perfctr = perfctr_attach_rdwr(child_pid, &state.control);
    if( !perfctr )
	perror("perfctr_attach_rdwr");
    close(pipe_to_child);		/* tell child 'go' */
    waitpid(child_pid, NULL, 0);
    if( !perfctr )
	return 1;
    if( perfctr_read(perfctr, &state) < 0 ) {
	perror("perfctr_read");
	return 1;
    }
    printf("\nFinal Sample:\n");
    printf("status\t\t\t%d\n", state.status);
    if( nrctrs > 1 )
	printf("control.evntsel[0]\t0x%08X\n", state.control.evntsel[0]);
    printf("tsc\t\t\t0x%016llX\n", state.sum.ctr[0]);
    if( nrctrs > 1 )
	printf("pmc[0]\t\t\t0x%016llX\n", state.sum.ctr[1]);
    return 0;
}

int main(int argc, char **argv)
{
    int pid;
    int pipe_fds[2];	/* for syncronising parent and child */

    if( argc < 2 ) {
	fprintf(stderr, "no command to run?\n");
	return 1;
    }
    init();
    if( pipe(pipe_fds) < 0 ) {
	perror("pipe");
	return 1;
    }
    if( (pid = fork()) < 0 ) {
	perror("fork");
	return 1;
    } else if( pid == 0 ) {
	close(pipe_fds[1]);
	return child(pipe_fds[0], argv+1);
    } else {
	close(pipe_fds[0]);
	return parent(pipe_fds[1], pid);
    }
}

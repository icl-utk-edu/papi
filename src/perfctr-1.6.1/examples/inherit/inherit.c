/* $Id$
 * inherit.c
 *
 * Illustrates perfctr inheritance from parent to child processes.
 *
 * Copyright (C) 2000  Mikael Pettersson
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include "libperfctr.h"

static unsigned nrctrs; /* > 1 if we have PMCs in addition to a TSC */

struct vperfctr *init_parent(const struct perfctr_dev *dev)
{
    struct perfctr_control control;
    struct vperfctr *perfctr;
    struct perfctr_info info;

    perfctr_info(dev, &info);
    nrctrs = perfctr_cpu_nrctrs(&info);
    perfctr = vperfctr_attach(dev);
    if( !perfctr ) {
	perror("parent: vperfctr_attach");
	exit(1);
    }
    memset(&control, 0, sizeof control);
    if( nrctrs > 1 )
	control.evntsel[0] = perfctr_evntsel_num_insns(&info);
    if( vperfctr_control(perfctr, &control) < 0 ) {
	perror("perror: vperfctr_control");
	exit(1);
    }
    return perfctr;
}

static void print_perfctr(const char *who, struct vperfctr *perfctr)
{
    struct vperfctr_state state;

    /* Stopping one's perfctrs is done here so we can verify that
     * the parent's children counters is the sum of its children's
     * final counters, as reported by this procedure.
     */
    if( vperfctr_stop(perfctr) < 0 )
	perror("vperfctr_stop");

    if( vperfctr_read_state(perfctr, &state) < 0 ) {
	perror("vperfctr_read_state");
	return;
    }
    printf("\nNear-Final Sample for %s:\n", who);
    printf("status\t\t\t%d\n", state.status);
    printf("control_id\t\t%u\n", state.control_id);
    if( nrctrs > 1 )
	printf("control.evntsel[0]\t0x%08X\n", state.control.evntsel[0]);
    printf("tsc\t\t\t0x%016llX\n", state.sum.ctr[0]);
    if( nrctrs > 1 )
	printf("pmc[0]\t\t\t0x%016llX\n", state.sum.ctr[1]);
    printf("children.tsc\t\t0x%016llX\n", state.children.ctr[0]);
    if( nrctrs > 1 )
	printf("children.pmc[0]\t\t0x%016llX\n", state.children.ctr[1]);
}

unsigned fac(unsigned n)
{
    return (n < 2) ? 1 : n * fac(n-1);
}

static void do_grandchild(const struct perfctr_dev *dev)
{
    struct vperfctr *grandchild;

    fac(20); /* work for a while */
    grandchild = vperfctr_attach(dev);
    if( !grandchild )
	perror("grandchild: vperfctr_attach");
    else
	print_perfctr("Grandchild", grandchild);
}

static void do_child(struct vperfctr *child, int grandchild_pid)
{
    fac(10); /* work for a while */
    waitpid(grandchild_pid, NULL, 0);
    print_perfctr("Child", child);
}

static void do_parent(struct vperfctr *parent, int child_pid)
{
    fac(5); /* work for a while */
    waitpid(child_pid, NULL, 0);
    print_perfctr("Parent", parent);
}

int main(void)
{
    struct perfctr_dev *dev;
    struct vperfctr *parent;
    int pid;

    dev = perfctr_dev_open();
    if( !dev ) {
	perror("perfctr_dev_open");
	return 1;
    }
    parent = init_parent(dev);
    if( (pid = fork()) < 0 ) {
	perror("fork");
	return 1;
    } else if( pid == 0 ) {
	struct vperfctr *child;
	child = vperfctr_attach(dev);
	if( !child ) {
	    perror("child: vperfctr_attach");
	    exit(1);
	}
	if( (pid = fork()) < 0 ) {
	    perror("fork");
	    return 1;
	} else if( pid == 0 ) {
	    do_grandchild(dev);
	} else {
	    do_child(child, pid);
	}
    } else {
	do_parent(parent, pid);
    }
    return 0;
}

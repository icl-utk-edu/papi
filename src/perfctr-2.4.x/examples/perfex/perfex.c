/* $Id$
 * perfex.c
 *
 * NAME
 *	perfex - a command-line interface to x86 performance counters
 *
 * SYNOPSIS
 *	perfex [-e event] .. [--p4pe=value] [--p4pmv=value] [-o file] command
 *	perfex { -i | -l | -L }
 *
 * DESCRIPTION
 *	The given command is executed; after it is complete, perfex
 *	prints the values of the various hardware performance counters.
 *
 * OPTIONS
 *	-e event | --event=event
 *		Specify an event to be counted.
 *		Multiple event specifiers may be given, limited by the
 *		number of available performance counters in the processor.
 *
 *		The full syntax of an event specifier is "evntsel/aux@pmc".
 *		All three components are 32-bit processor-specific numbers,
 *		written in decimal or hexadecimal notation.
 *
 *		"evntsel" is the primary processor-specific event selection
 *		code to use for this event. This field is mandatory.
 *
 *		"/aux" is used when additional event selection data is
 *		needed. For the Pentium 4, "evntsel" is put in the counter's
 *		CCCR register, and "aux" is put in the associated ESCR
 *		register. No other processor currently needs this field.
 *
 *		"@pmc" describes which CPU counter number to assign this
 *		event to. When omitted, the events are assigned in the
 *		order listed, starting from 0. Either all or none of the
 *		event specifiers should use the "@pmc" notation.
 *		Explicit counter assignment via "@pmc" is required on
 *		Pentium 4 and VIA C3 processors.
 *
 *		The counts, together with an event description are written
 *		to the result file (default is stderr).
 *
 *	--p4pe=value | --p4_pebs_enable=value
 *	--p4pmv=value | --p4_pebs_matrix_vert=value
 *		Specify the value to be stored in the auxiliary control
 *		register PEBS_ENABLE or PEBS_MATRIX_VERT, which are used
 *		for replay tagging events on Pentium 4 processors.
 *		Note: Intel's documentation states that bit 25 should be
 *		set in PEBS_ENABLE, but this is not true and the driver
 *		will disallow it.
 *
 *	-i | --info
 *		Instead of running a command, generate output which
 *		identifies the current processor and its capabilities.
 *
 *	-l | --list
 *		Instead of running a command, generate output which
 *		identifies the current processor and its capabilities,
 *		and lists its countable events.
 *
 *	-L | --long-list
 *		Like -l, but list the events in a more detailed format.
 *
 *	-o file | --output=file
 *		Write the results to file instead of stderr.
 *
 * EXAMPLES
 *	The following commands count the number of retired instructions
 *	in user-mode on an Intel P6 processor:
 *
 *	perfex -e 0x004100C0 some_program
 *	perfex --event=0x004100C0 some_program
 *
 *	The following command does the same on an Intel Pentium 4 processor:
 *
 *	perfex -e 0x00039000/0x04000204@0x8000000C some_program
 *
 *	Explanation: Program IQ_CCCR0 with required flags, ESCR select 4
 *	(== CRU_ESCR0), and Enable. Program CRU_ESCR0 with event 2
 *	(instr_retired), NBOGUSNTAG, CPL>0. Map this event to IQ_COUNTER0
 *	(0xC) with fast RDPMC enabled.
 *
 *	The following command counts the number of L1 cache read misses
 *	on a Pentium 4 processor:
 *
 *	perfex -e 0x0003B000/0x12000204@0x8000000C --p4pe=0x01000001 --p4pmv=0x1 some_program
 *
 *	Explanation: IQ_CCCR0 is bound to CRU_ESCR2, CRU_ESCR2 is set up
 *	for replay_event with non-bogus uops and CPL>0, and PEBS_ENABLE
 *	and PEBS_MATRIX_VERT are set up for the 1stL_cache_load_miss_retired
 *	metric. Note that bit 25 is NOT set in PEBS_ENABLE.
 *
 * DEPENDENCIES
 *	perfex only works on Linux/x86 systems which have been modified
 *	to include the perfctr driver. This driver is available at
 *	http://www.csd.uu.se/~mikpe/linux/perfctr/.
 *
 * BUGS
 *	The -l and -L options do not work on the Pentium 4.
 *
 * NOTES
 *	perfex is superficially similar to IRIX' perfex(1).
 *	The -a, -mp, -s, and -x options are not yet implemented.
 *
 * Copyright (C) 1999-2002  Mikael Pettersson
 */

/*
 * Theory of operation:
 * - Parent creates a socketpair().
 * - Parent forks.
 * - Child opens /proc/self/perfctr and sets up its perfctrs.
 * - Child sends its perfctr fd to parent via the socketpair().
 * - Child exec:s the command.
 * - Parent waits for child to exit.
 * - Parent receives child's perfctr fd via the socketpair().
 * - Parent mmap():s child's perfctr fd and reads the final counts.
 */

#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>	/* for strerror() */
#include <unistd.h>
#include "libperfctr.h"
#define PAGE_SIZE	4096

#define ARRAY_SIZE(x)	(sizeof(x) / sizeof((x)[0]))

/*
 * Our child-to-parent protocol is the following:
 * There is an int-sized data packet, with an optional 'struct cmsg_fd'
 * control message attached.
 * The data packet (which must be present, as control messages don't
 * work with zero-sized payloads) contains an 'int' status.
 * If status != 0, then it is an 'errno' value from the child's
 * perfctr setup code.
 */

struct cmsg_fd {
    struct cmsghdr hdr;
    int fd;
};

static int my_send(int sock, int fd, int status)
{
    struct msghdr msg;
    struct iovec iov;
    struct cmsg_fd cmsg_fd;
    int buf[1];

    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    msg.msg_flags = 0;

    buf[0] = status;
    iov.iov_base = buf;
    iov.iov_len = sizeof buf;
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    if( status != 0 ) {	/* errno, don't send fd */
	msg.msg_control = 0;
	msg.msg_controllen = 0;
    } else {
	cmsg_fd.hdr.cmsg_len = sizeof cmsg_fd;
	cmsg_fd.hdr.cmsg_level = SOL_SOCKET;
	cmsg_fd.hdr.cmsg_type = SCM_RIGHTS;
	cmsg_fd.fd = fd;
	msg.msg_control = &cmsg_fd;
	msg.msg_controllen = sizeof cmsg_fd;
    }
    return sendmsg(sock, &msg, 0) == sizeof buf ? 0 : -1;
}

static int my_send_fd(int sock, int fd)
{
    return my_send(sock, fd, 0);
}

static int my_send_err(int sock)
{
    return my_send(sock, -1, errno);
}

static int my_receive(int sock, int *fd)
{
    struct msghdr msg;
    struct iovec iov;
    struct cmsg_fd cmsg_fd;
    int buf[1];

    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    msg.msg_flags = 0;

    buf[0] = -1;
    iov.iov_base = buf;
    iov.iov_len = sizeof(buf);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    memset(&cmsg_fd, ~0, sizeof cmsg_fd);
    msg.msg_control = &cmsg_fd;
    msg.msg_controllen = sizeof cmsg_fd;

    if( recvmsg(sock, &msg, 0) != sizeof buf )
	return -1;

    if( buf[0] == 0 &&
	msg.msg_control == &cmsg_fd &&
	msg.msg_controllen == sizeof cmsg_fd &&
	cmsg_fd.hdr.cmsg_type == SCM_RIGHTS &&
	cmsg_fd.hdr.cmsg_level == SOL_SOCKET &&
	cmsg_fd.hdr.cmsg_len == sizeof cmsg_fd &&
	cmsg_fd.fd >= 0 ) {
	*fd = cmsg_fd.fd;
	return 0;
    }

    if( msg.msg_controllen == 0 && buf[0] != 0 )
	errno = buf[0];
    else
	errno = EPROTO;
    return -1;
}

static int do_child(int sock, const struct vperfctr_control *control, char **argv)
{
    int fd;

    fd = open("/proc/self/perfctr", O_RDONLY|O_CREAT);
    if( fd < 0 ) {
	my_send_err(sock);
	return 1;
    }
    if( ioctl(fd, VPERFCTR_CONTROL, control) < 0 ) {
	my_send_err(sock);
	return 1;
    }
    if( my_send_fd(sock, fd) < 0 ) {
	my_send_err(sock);	/* well, we can try.. */
	return 1;
    }
    close(fd);
    close(sock);
    execvp(argv[0], argv);
    perror(argv[0]);
    return 1;
}

static int do_parent(int sock, int child_pid, FILE *resfile)
{
    int child_status;
    int fd, i;
    int nrctrs;
    volatile const struct vperfctr_state *kstate;

    /* this can be done before or after the recvmsg() */
    if( waitpid(child_pid, &child_status, 0) < 0 ) {
	perror("perfex: waitpid");
	return 1;
    }
    if( !WIFEXITED(child_status) ) {
	fprintf(stderr, "perfex: child did not exit normally\n");
	return 1;
    }
    if( my_receive(sock, &fd) < 0 ) {
	perror("perfex: receiving fd/status");
	return 1;
    }
    close(sock);
    kstate = mmap(NULL, PAGE_SIZE, PROT_READ, MAP_SHARED, fd, 0);
    if( kstate == MAP_FAILED ) {
	perror("perfex: mmap");
	return 1;
    }
    if( kstate->magic != VPERFCTR_MAGIC ) {
	fprintf(stderr, "perfex: kstate version mismatch, kernel %#x, expected %#x\n",
		kstate->magic, VPERFCTR_MAGIC);
	return 1;
    }
    close(fd);

    if( kstate->cpu_state.control.tsc_on )
	fprintf(resfile, "tsc\t\t\t%19lld\n", kstate->cpu_state.sum.tsc);
    nrctrs = kstate->cpu_state.control.nractrs;
    for(i = 0; i < nrctrs; ++i) {
	fprintf(resfile, "event 0x%08X",
		kstate->cpu_state.control.evntsel[i]);
	if( kstate->cpu_state.control.evntsel_aux[i] )
	    fprintf(resfile, "/0x%08X",
		    kstate->cpu_state.control.evntsel_aux[i]);
	fprintf(resfile, "\t%19lld\n", kstate->cpu_state.sum.pmc[i]);
    }
    if( kstate->cpu_state.control.p4.pebs_enable )
	fprintf(resfile, "PEBS_ENABLE 0x%08X\n",
		kstate->cpu_state.control.p4.pebs_enable);
    if( kstate->cpu_state.control.p4.pebs_matrix_vert )
	fprintf(resfile, "PEBS_MATRIX_VERT 0x%08X\n",
		kstate->cpu_state.control.p4.pebs_matrix_vert);

    munmap((void*)kstate, PAGE_SIZE);
    return WEXITSTATUS(child_status);
}

static int do_perfex(const struct vperfctr_control *control, char **argv, FILE *resfile)
{
    int pid;
    int sv[2];

    if( socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) < 0 ) {
	perror("perfex: socketpair");
	return 1;
    }
    pid = fork();
    if( pid < 0 ) {
	perror("perfex: fork");
	return 1;
    }
    if( pid == 0 ) {
	close(sv[0]);
	return do_child(sv[1], control, argv);
    } else {
	close(sv[1]);
	return do_parent(sv[0], pid, resfile);
    }
}

static int get_info(struct perfctr_info *info)
{
    int fd;

    fd = open("/proc/self/perfctr", O_RDONLY);
    if( fd < 0 ) {
	perror("perfex: open /proc/self/perfctr");
	return -1;
    }
    if( ioctl(fd, PERFCTR_INFO, info) != 0 ) {
	perror("perfex: PERFCTR_INFO");
	close(fd);
	return -1;
    }
    close(fd);
    return 0;
}

static void do_print_event(const struct perfctr_event *event, int long_format)
{
    printf("%s", event->name);
    if( long_format )
	printf(":0x%02X:0x%X:0x%X",
	       event->code,
	       event->counters_mask,
	       event->default_qualifier);
    printf("\n");
}

static void do_print_event_set(const struct perfctr_event_set *event_set,
			       int long_format)
{
    unsigned int i;

    if( event_set->include )
	do_print_event_set(event_set->include, long_format);
    for(i = 0; i < event_set->nevents; ++i)
	do_print_event(&event_set->events[i], long_format);
}

static int do_list(const struct perfctr_info *info, int long_format)
{
    const struct perfctr_event_set *event_set;
    unsigned int nrctrs;

    event_set = perfctr_cpu_event_set(info->cpu_type);
    if( !event_set ) {
	fprintf(stderr, "perfex: perfctr_cpu_event_set(%u) failed\n",
		info->cpu_type);
	return 1;
    }
    printf("CPU type %s\n", perfctr_cpu_name(info));
    printf("%s time-stamp counter available\n",
	   (info->cpu_features & PERFCTR_FEATURE_RDTSC) ? "One" : "No");
    nrctrs = perfctr_cpu_nrctrs(info);
    printf("%u performance counter%s available\n",
	   nrctrs, (nrctrs == 1) ? "" : "s");
    if( !event_set->nevents ) /* the 'generic' CPU type */
	return 0;
    printf("\nAvailable Events:\n");
    if( long_format )
	printf("Name:Code:CounterMask:DefaultQualifier\n");
    do_print_event_set(event_set, long_format);
    return 0;
}

static const struct option long_options[] = {
    { "event", 1, NULL, 'e' },
    { "p4pe", 1, NULL, 1 }, { "p4_pebs_enable", 1, NULL, 1 },
    { "p4pmv", 1, NULL, 2 }, { "p4_pebs_matrix_vert", 1, NULL, 2 },
    { "info", 0, NULL, 'i' },
    { "list", 0, NULL, 'l' },
    { "long-list", 0, NULL, 'L' },
    { "output", 1, NULL, 'o' },
    { 0 }
};

static void do_usage(void)
{
    fprintf(stderr, "Usage:  perfex [options] <command> [<command arg>] ...\n");
    fprintf(stderr, "\tperfex -i\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "\t-e <event> | --event=<event>\tEvent to be counted\n");
    fprintf(stderr, "\t-o <file> | --output=<file>\tWrite output to file (default is stderr)\n");
    fprintf(stderr, "\t-i | --info\t\t\tPrint PerfCtr driver information\n");
    fprintf(stderr, "\t-l | --list\t\t\tList available events\n");
    fprintf(stderr, "\t-L | --long-list\t\tList available events in long format\n");
    fprintf(stderr, "\t--p4pe=<value>\t\t\tValue for PEBS_ENABLE (P4 only)\n");
    fprintf(stderr, "\t--p4_pebs_enable=<value>\tSame as --p4pe=<value>\n");
    fprintf(stderr, "\t--p4pmv=<value>\t\t\tValue for PEBS_MATRIX_VERT (P4 only)\n");
    fprintf(stderr, "\t--p4_pebs_matrix_vert=<value>\tSame as --p4pmv=<value>\n");
}

static int parse_event_spec(const char *arg, unsigned int *evntsel,
			    unsigned int *aux, unsigned int *pmc)
{
    char *endp;

    *evntsel = strtoul(arg, &endp, 16);
    if( endp[0] != '/' ) {
	*aux = 0;
    } else {
	arg = endp + 1;
	*aux = strtoul(arg, &endp, 16);
    }
    if( endp[0] != '@' ) {
	*pmc = (unsigned int)-1;
    } else {
	arg = endp + 1;
	*pmc = strtoul(arg, &endp, 16);
    }
    return endp[0] != '\0';
}

static int parse_value(const char *arg, unsigned int *value)
{
    char *endp;

    *value = strtoul(arg, &endp, 16);
    return endp[0] != '\0';
}

int main(int argc, char **argv)
{
    struct perfctr_info info;
    struct vperfctr_control control;
    int n;
    unsigned int spec_evntsel, spec_aux, spec_pmc;
    unsigned int spec_value;
    FILE *resfile;

    /* prime info, as we'll need it in most cases */
    if( get_info(&info) )
	return 1;

    memset(&control, 0, sizeof control);
    if( info.cpu_features & PERFCTR_FEATURE_RDTSC )
	control.cpu_control.tsc_on = 1;
    n = 0;
    resfile = stderr;

    for(;;) {
	/* the '+' is there to prevent permutation of argv[] */
	switch( getopt_long(argc, argv, "+e:ilLo:", long_options, NULL) ) {
	  case -1:	/* no more options */
	    if( optind >= argc ) {
		fprintf(stderr, "perfex: command missing\n");
		return 1;
	    }
	    argv += optind;
	    break;
	  case 'i':
	    printf("PerfCtr Info:\n");
	    perfctr_print_info(&info);
	    return 0;
	  case 'l':
	    return do_list(&info, 0);
	  case 'L':
	    return do_list(&info, 1);
	  case 'o':
	    if( (resfile = fopen(optarg, "w")) == NULL ) {
		fprintf(stderr, "perfex: %s: %s\n", optarg, strerror(errno));
		return 1;
	    }
	    continue;
	  case 'e':
	    if( parse_event_spec(optarg, &spec_evntsel, &spec_aux, &spec_pmc) ) {
		fprintf(stderr, "perfex: invalid event specifier: '%s'\n", optarg);
		return 1;
	    }
	    if( n >= ARRAY_SIZE(control.cpu_control.evntsel) ) {
		fprintf(stderr, "perfex: too many event specifiers\n");
		return 1;
	    }
	    if( spec_pmc == (unsigned int)-1 )
		spec_pmc = n;
	    control.cpu_control.evntsel[n] = spec_evntsel;
	    control.cpu_control.evntsel_aux[n] = spec_aux;
	    control.cpu_control.pmc_map[n] = spec_pmc;
	    control.cpu_control.nractrs = ++n;
	    continue;
	  case 1:
	    if( parse_value(optarg, &spec_value) ) {
		fprintf(stderr, "perfex: invalid value: '%s'\n", optarg);
		return 1;
	    }
	    control.cpu_control.p4.pebs_enable = spec_value;
	    continue;
	  case 2:
	    if( parse_value(optarg, &spec_value) ) {
		fprintf(stderr, "perfex: invalid value: '%s'\n", optarg);
		return 1;
	    }
	    control.cpu_control.p4.pebs_matrix_vert = spec_value;
	    continue;
	  default:
	    do_usage();
	    return 1;
	}
	break;
    }

    return do_perfex(&control, argv, resfile);
}

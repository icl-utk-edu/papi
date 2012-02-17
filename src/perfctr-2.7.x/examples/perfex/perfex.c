/* $Id: perfex.c,v 1.35 2005/01/16 22:51:20 mikpe Exp $
 *
 * NAME
 *	perfex - a command-line interface to processor performance counters
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
 *		The full syntax of an event specifier is "evntsel/escr@pmc".
 *		All three components are 32-bit processor-specific numbers,
 *		written in decimal or hexadecimal notation.
 *
 *		"evntsel" is the primary processor-specific event selection
 *		code to use for this event. This field is mandatory.
 *
 *		"/escr" is used to specify additional event selection data
 *		for Pentium 4 processors. "evntsel" is put in the counter's
 *		CCCR register, and "escr" is put in the associated ESCR
 *		register.
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
 *	perfex only works on Linux systems which have been modified
 *	to include the perfctr kernel extension. Perfctr is available at
 *	http://www.csd.uu.se/~mikpe/linux/perfctr/.
 *
 * NOTES
 *	perfex is superficially similar to IRIX' perfex(1).
 *	The -a, -mp, -s, and -x options are not yet implemented.
 *
 * Copyright (C) 1999-2004  Mikael Pettersson
 */

/*
 * Theory of operation:
 * - Parent creates a socketpair().
 * - Parent forks.
 * - Child creates and sets up its perfctrs.
 * - Child sends its perfctr fd to parent via the socketpair().
 * - Child exec:s the command.
 * - Parent waits for child to exit.
 * - Parent receives child's perfctr fd via the socketpair().
 * - Parent retrieves child's final control and counts via the fd.
 */

#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <errno.h>
#include <getopt.h>
#include <stddef.h>	/* for offsetof() */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>	/* for strerror() */
#include <unistd.h>
#include "libperfctr.h"
#include "arch.h"

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
    /* 64-bit machines pad here, which causes problems since
       the kernel derives the number of fds from the size.
       The CMSG_FD_TRUE_SIZE macro gives the true payload size. */
};
#define CMSG_FD_TRUE_SIZE	(offsetof(struct cmsg_fd, fd) + sizeof(int))
#define CMSG_FD_PADDED_SIZE	sizeof(struct cmsg_fd)

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
	cmsg_fd.hdr.cmsg_len = CMSG_FD_TRUE_SIZE;
	cmsg_fd.hdr.cmsg_level = SOL_SOCKET;
	cmsg_fd.hdr.cmsg_type = SCM_RIGHTS;
	cmsg_fd.fd = fd;
	msg.msg_control = &cmsg_fd;
	msg.msg_controllen = CMSG_FD_TRUE_SIZE;
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
    msg.msg_controllen = CMSG_FD_TRUE_SIZE;

    if( recvmsg(sock, &msg, 0) != sizeof buf )
	return -1;

    if( buf[0] == 0 &&
	msg.msg_control == &cmsg_fd &&
	msg.msg_controllen == CMSG_FD_PADDED_SIZE &&
	cmsg_fd.hdr.cmsg_type == SCM_RIGHTS &&
	cmsg_fd.hdr.cmsg_level == SOL_SOCKET &&
	cmsg_fd.hdr.cmsg_len == CMSG_FD_TRUE_SIZE &&
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

static int do_open_self(int creat)
{
    int fd;

    fd = _vperfctr_open(creat);
    if( fd >= 0 && perfctr_abi_check_fd(fd) < 0 ) {
	close(fd);
	return -1;
    }
    return fd;
}

static int do_child(int sock, const struct vperfctr_control *control, char **argv)
{
    int fd;

    fd = do_open_self(1);
    if( fd < 0 ) {
	my_send_err(sock);
	return 1;
    }
    if( _vperfctr_control(fd, control) < 0 ) {
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
    int fd;
    struct perfctr_sum_ctrs sum;
    struct vperfctr_control control;
    struct perfctr_sum_ctrs children;

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
    /* XXX: surely we don't need to repeat the ABI check here? */
    if( _vperfctr_read_sum(fd, &sum) < 0 ) {
	perror("perfex: read_sum");
	return 1;
    }
    if( _vperfctr_read_control(fd, &control) < 0 ) {
	perror("perfex: read_control");
	return 1;
    }
    if( _vperfctr_read_children(fd, &children) < 0 ) {
	perror("perfex: read_children");
	return 1;
    }
    close(fd);

    do_print(resfile, &control.cpu_control, &sum, &children);

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

    fd = do_open_self(0);
    if( fd < 0 ) {
	perror("perfex: open perfctrs");
	return -1;
    }
    if( perfctr_info(fd, info) < 0 ) {
	perror("perfex: perfctr_info");
	close(fd);
	return -1;
    }
    close(fd);
    return 0;
}

static struct perfctr_cpus_info *get_cpus_info(void)
{
    int fd;
    struct perfctr_cpus_info *cpus_info;

    fd = do_open_self(0);
    if( fd < 0 ) {
	perror("perfex: open perfctrs");
	return NULL;
    }
    cpus_info = perfctr_cpus_info(fd);
    if( !cpus_info )
	perror("perfex: perfctr_cpus_info");
    close(fd);
    return cpus_info;
}

static int do_info(const struct perfctr_info *info)
{
    struct perfctr_cpus_info *cpus_info;

    cpus_info = get_cpus_info();
    printf("PerfCtr Info:\n");
    perfctr_info_print(info);
    if( cpus_info ) {
	perfctr_cpus_info_print(cpus_info);
	free(cpus_info);
    }
    return 0;
}

static void do_print_event(const struct perfctr_event *event, int long_format,
			   const char *event_prefix)
{
    printf("%s%s", event_prefix, event->name);
    if( long_format )
	printf(":0x%02X:0x%X:0x%X",
	       event->evntsel,
	       event->counters_set,
	       event->unit_mask ? event->unit_mask->default_value : 0);
    printf("\n");
}

static void do_print_event_set(const struct perfctr_event_set *event_set,
			       int long_format)
{
    unsigned int i;

    if( event_set->include )
	do_print_event_set(event_set->include, long_format);
    for(i = 0; i < event_set->nevents; ++i)
	do_print_event(&event_set->events[i], long_format, event_set->event_prefix);
}

static int do_list(const struct perfctr_info *info, int long_format)
{
    const struct perfctr_event_set *event_set;
    unsigned int nrctrs;

    printf("CPU type %s\n", perfctr_info_cpu_name(info));
    printf("%s time-stamp counter available\n",
	   (info->cpu_features & PERFCTR_FEATURE_RDTSC) ? "One" : "No");
    nrctrs = perfctr_info_nrctrs(info);
    printf("%u performance counter%s available\n",
	   nrctrs, (nrctrs == 1) ? "" : "s");
    printf("Overflow interrupts%s available\n",
	   (info->cpu_features & PERFCTR_FEATURE_PCINT) ? "" : " not");

    event_set = perfctr_cpu_event_set(info->cpu_type);
    if( !event_set ) {
	fprintf(stderr, "perfex: perfctr_cpu_event_set(%u) failed\n",
		info->cpu_type);
	return 1;
    }
    if( !event_set->nevents ) /* the 'generic' CPU type */
	return 0;
    printf("\nEvents Available:\n");
    if( long_format )
	printf("Name:EvntSel:CounterSet:DefaultUnitMask\n");
    do_print_event_set(event_set, long_format);
    return 0;
}

/* Hack while phasing out an old number parsing bug. */
static unsigned int strtoul_base = 16;
static unsigned int quiet;

unsigned long my_strtoul(const char *nptr, char **endptr)
{
    unsigned long val1;

    val1 = strtoul(nptr, endptr, strtoul_base);
    if (strtoul_base == 16 && !quiet) {
	unsigned long val2 = strtoul(nptr, NULL, 0);
	if (val1 != val2)
	    fprintf(stderr, "perfex: warning: string '%s' is base-dependent, assuming base 16."
		    " Please prefix hexadecimal numbers with '0x'.\n",
		    nptr);
    }
    return val1;
}

static const struct option long_options[] = {
    { "decimal", 0, NULL, 'd' },
    { "event", 1, NULL, 'e' },
    { "help", 0, NULL, 'h' },
    { "hex", 0, NULL, 'x' },
    { "info", 0, NULL, 'i' },
    { "list", 0, NULL, 'l' },
    { "long-list", 0, NULL, 'L' },
    { "output", 1, NULL, 'o' },
    ARCH_LONG_OPTIONS
    { 0 }
};

static void do_usage(void)
{
    fprintf(stderr, "Usage:  perfex [options] <command> [<command arg>] ...\n");
    fprintf(stderr, "\tperfex -i\n");
    fprintf(stderr, "\tperfex -h\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "\t-e <event> | --event=<event>\tEvent to be counted\n");
    fprintf(stderr, "\t-h | --help\t\t\tPrint this help text\n");
    fprintf(stderr, "\t-o <file> | --output=<file>\tWrite output to file (default is stderr)\n");
    fprintf(stderr, "\t-i | --info\t\t\tPrint PerfCtr driver information\n");
    fprintf(stderr, "\t-l | --list\t\t\tList available events\n");
    fprintf(stderr, "\t-L | --long-list\t\tList available events in long format\n");
    fprintf(stderr, "\t-d | --decimal\t\t\tAllow decimal numbers in event specifications\n");
    fprintf(stderr, "\t-x | --hex\t\t\tOnly accept hexadecimal numbers in event specifications\n");
    do_arch_usage();
}

int main(int argc, char **argv)
{
    struct perfctr_info info;
    struct vperfctr_control control;
    int n;
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
	int ch = getopt_long(argc, argv, "+de:hilLo:x", long_options, NULL);
	switch( ch ) {
	  case -1:	/* no more options */
	    if( optind >= argc ) {
		fprintf(stderr, "perfex: command missing\n");
		return 1;
	    }
	    argv += optind;
	    break;
	  case 'h':
	    do_usage();
	    return 0;
	  case 'i':
	    return do_info(&info);
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
	  case 'd':
	    strtoul_base = 0;
	    continue;
	  case 'x':
	    strtoul_base = 16;
	    quiet = 1;
	    continue;
	  case 'e':
	    n = do_event_spec(n, optarg, &control.cpu_control);
	    continue;
	  default:
	    if( do_arch_option(ch, optarg, &control.cpu_control) < 0 ) {
		do_usage();
		return 1;
	    }
	    continue;
	}
	break;
    }

    return do_perfex(&control, argv, resfile);
}

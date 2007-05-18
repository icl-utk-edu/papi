#!/bin/sh
#
#	(C) COPYRIGHT CRAY INC.
#	UNPUBLISHED PROPRIETARY INFORMATION.
#	ALL RIGHTS RESERVED.
#
#	Transform the ASCII perfmon events file into a static table.

# print "#define STATIC_PERFMON_EVENTS_TABLE 1"
echo "static char *perfmon_events_table ="
cat perfmon_events.csv | \
	tr "\"" "'" |
	sed 's/^/"/' | \
	sed 's/$/\\n\"/'
echo ";"

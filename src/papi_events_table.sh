#!/bin/sh
#
#	(C) COPYRIGHT CRAY INC.
#	UNPUBLISHED PROPRIETARY INFORMATION.
#	ALL RIGHTS RESERVED.
#
#	Transform the papi_events.csv file into a static table.

# print "#define STATIC_PAPI_EVENTS_TABLE 1"
echo "static char *papi_events_table ="
cat $1 | \
	tr "\"" "'" |
	sed 's/^/"/' | \
	sed 's/$/\\n\"/'
echo ";"

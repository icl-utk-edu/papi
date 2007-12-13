#!/bin/sh
# print "#define STATIC_PERFMON_EVENTS_TABLE 1"
echo "static char *perfmon_events_table ="
cat perfmon_events.csv | \
	tr "\"" "'" |
	grep -v "^#" | \
	sed 's/^/"/' | \
	sed 's/$/\\n\"/'
echo ";"

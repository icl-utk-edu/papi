#!/bin/sh
#
# Simplictic cycle breakdown generator
#
# Copyright (C) 2001 Hewlett-Packard Co
# Contributed by Stephane Eranian <eranian@hpl.hp.com>
#
usage()
{
	echo "Usage $0 [-k] [-u] [-h] [-i cmd_redir_in] [-o cmd_redir_out] command [command_args]"
}

TMPDIR=/tmp

dot=$(dirname $0)

tmp=`getopt -o hkui:o: -- "$@"`

if [ $? -ne 0 ]; then    
	echo  >&2 "Usage: $0 [-k][-u] program"
	exit 1
fi

eval set -- "$tmp"

user=""
kernel=""
inp=""
output=""

while true
do 
   case "$1" in
   -u)     shift; user="--user-level";;
   -k)     shift; kernel="--kernel-level";;
   -i)     input="$2"; shift 2;;
   -o)     output="$2"; shift 2;;
   -h)	   usage; exit 0;;
   --)     shift; break;;
   esac
done

if [ -z "$1" ]; then
	echo you need to specify a command
	exit 1
fi

redir_in=${input:-"/dev/null"}
redir_out=${output:-"/dev/null"}

today=`date +'%y%m%d'`
RESFILE=$TMPDIR/`basename $1`-$today

PFMON_OPT="$user $kernel --append --outfile=$RESFILE"
PFMON_CMD=pfmon
PFMON="$PFMON_CMD $PFMON_OPT"
echo "arguments to pfmon $PFMON_OPT"
CMD="-- $*"

echo Command is :$CMD:

rm -rf $RESFILE

$PFMON -e CPU_CYCLES,PIPELINE_BACKEND_FLUSH_CYCLE,PIPELINE_ALL_FLUSH_CYCLE $CMD < $redir_in > $redir_out
$PFMON -e CPU_CYCLES,UNSTALLED_BACKEND_CYCLE,INST_ACCESS_CYCLE $CMD < $redir_in > $redir_out
$PFMON -e CPU_CYCLES,DEPENDENCY_SCOREBOARD_CYCLE,DEPENDENCY_ALL_CYCLE $CMD < $redir_in >$redir_out
$PFMON -e CPU_CYCLES,MEMORY_CYCLE,DATA_ACCESS_CYCLE $CMD < $redir_in  > $redir_out

echo "--> Results saved in $RESFILE <--"

cat >/tmp/cyres$$.py <<"EOF"
#!/usr/bin/python
#
# Copyright (C) 2001 Hewlett-Packard Co
# Contributed by Stephane Eranian <eranian@hpl.hp.com>
#
import sys,string

#
# error printing routine
#
def error(fmt, *args):
	sys.stderr.write(fmt % args)
	sys.exit(1)

# return true if element is not the empty element
# used to filter out empty entries in the list
#
def empty(l):  return l!=''


#
# main function
#	
if len(sys.argv) > 1: 
	f = open(sys.argv[1], 'r')
else:
	f = sys.stdin

evt={'':0}
cycle_smpl=[0,0,0,0]
cycles=0
k=0
for l in f.readlines():

	# skip comment lines
	if l[0]=='#': continue

	# transform into list of elements
	p = string.split(l[:-1], ' ')

	# filter out '' elements
	p = filter(empty,p)

	if p[1] == 'CPU_CYCLES' : 
		cycle_smpl[k] = string.atol(p[0])
		k = k + 1
		cycles = cycles + string.atol(p[0])
	else:
		evt[p[1]] = string.atol(p[0])
#
# now do pretty printing
# 
#tot_cycles=evt['CPU_CYCLES']
tot_cycles=cycles/k

evt['CPU_CYCLES'] = tot_cycles
sum=0
sum = evt['DEPENDENCY_ALL_CYCLE'] + evt['MEMORY_CYCLE'] + evt['UNSTALLED_BACKEND_CYCLE']+evt['PIPELINE_ALL_FLUSH_CYCLE']

print "0. cycles=%lu comb=%lu diff=%5.2f%%\n" % (tot_cycles, sum, abs((sum-tot_cycles)*100/tot_cycles))
print "                                                cycles"
print '-'*70

p = 0

v=evt['DEPENDENCY_SCOREBOARD_CYCLE']
p = p + v
print "1. dependency cycles                          %12lu\t%5.2f%%" % (v, v*100.0/tot_cycles)

v=evt['DEPENDENCY_ALL_CYCLE']-evt['DEPENDENCY_SCOREBOARD_CYCLE']
p = p + v
print "2. issue limit cycles                         %12lu\t%5.2f%%" % (v, v*100.0/tot_cycles)


v=evt['DATA_ACCESS_CYCLE']
p = p + v
print "3. data access cycles                         %12lu\t%5.2f%%" % (v, v*100.0/tot_cycles)

v=evt['INST_ACCESS_CYCLE']
p = p + v
print "4. instruction access cycles                  %12lu\t%5.2f%%" % (v, v*100.0/tot_cycles)

v=evt['MEMORY_CYCLE']-evt['DATA_ACCESS_CYCLE']
p = p + v
print "5. RSE memory cycles                          %12lu\t%5.2f%%" % (v, v*100.0/tot_cycles)



v=evt['UNSTALLED_BACKEND_CYCLE']-evt['INST_ACCESS_CYCLE']
p = p + v
print "6. inherent execution cycles                  %12lu\t%5.2f%%" % (v, v*100.0/tot_cycles)

v=evt['PIPELINE_BACKEND_FLUSH_CYCLE']
p = p + v
print "7. branch resteer cycles                      %12lu\t%5.2f%%" % (v, v*100.0/tot_cycles)

v=evt['PIPELINE_ALL_FLUSH_CYCLE']-evt['PIPELINE_BACKEND_FLUSH_CYCLE']
p = p + v
print "8. taken branch cycles                        %12lu\t%5.2f%%" % (v, v*100.0/tot_cycles)

print '-'*70
print "                                              %12lu\t%5.2f%%" %(tot_cycles, p*100/tot_cycles)

EOF


python /tmp/cyres$$.py $RESFILE
rm /tmp/cyres$$.py

cat >/tmp/valid$$.awk <<"EOF"
#! /usr/bin/awk -f
#
# Copyright (C) 2001 Hewlett-Packard Co
# Contributed by Sumit Roy <sumit_roy@hp.com>
#
# awk program to validate the output from any program that produces two column
# output of the form:
#     DATA_NAME    value.
# The program will compute the standard deviation and mean of the values.

function mean(input_array, input_data, input_count, local_sum, i) 
{
   for (i = 1; i <= input_count; i++) {
      local_sum += input_array[input_data, i];
   }
   return local_sum/input_count;
}

function variance(input_array, input_data, input_count, input_mean, \
		  local_variance, i) 
{
   for (i = 1; i <= input_count; i++) {
      local_variance += (input_array[input_data, i] - input_mean) \
	 * (input_array[input_data, i] - input_mean) / input_count;
   }
   return local_variance;
}

function print_stats(input_array, input_data, input_count, \
		     local_mean, local_variance) 
{
   local_mean = mean(input_array, input_data, input_count);
   
   local_variance = variance(input_array, input_data, input_count, local_mean);
   printf "SD: %7.3f %% mean: %.3f\n", \
	  sqrt(local_variance)*100/local_mean,\
	  local_mean;
}

{
   count[$2] += 1;
   array[$2, count[$2]] = $1;
}


END {
# print the stats for each of the data items
   for (data in count) {
      printf "%-25s\t", data;      
      print_stats(array, data, count[data]);
   }
   
}
EOF

cat $RESFILE | awk -f /tmp/valid$$.awk
rm /tmp/valid$$.awk
exit 0

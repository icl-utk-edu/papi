#!/usr/bin/python
#
# genpme.py: generate pme_list.h from the perfmon database
#
# Copyright (C) 2001 Hewlett-Packard Co
# Contributed by Stephane Eranian <eranian@hpl.hp.com>
#
# This file is part of pfmon, a sample tool to measure performance 
# of applications on Linux/ia64.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# 02111-1307 USA
#

import sys,string

#
# global parameters
#

# list of implements counters(Itanium has only 4)
# use tuples for immutability
impl_counters=(4,5,6,7)

# line number in file
#
line=0

# list of supported qualifiers (tuples for immutability)
# Descriptions:
# 	I : Instruction Address Range
#	O : Opcode match
#	D : Data Address Range
#	- : none supported
#
qual_list=('I','O','D','-')


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
# returns the umask given the string umask 'xxxx' where x is 0 or 1 only
#
def gen_umask(p):

	# 
	# umask must be specified in binary base
	#
	val_lst=['0','1']
	val=0
	for i in range(3,-1,-1):
		if p[i] not in val_lst: 
			error('line %d: invalid umask 0x%x\n', line, p[i])

		val = val | val_lst.index(p[i])*pow(2,(3-i))
	return val

#
# generate a counter mask based on list of valid counter for event
# bit zero of mask corresponds to counter 4 (first counter)
#
def gen_counter(p):
	val=0

	for i in range(0,len(p)):
		s=string.split(p[i],'-')

		start=string.atoi(s[0])
		if len(s) > 1: 
			end=string.atoi(s[1])+1
		else:
			end=string.atoi(s[0])+1

		for j in range(start,end):
			if j not in impl_counters:
				error('line %d: invalid counter %d\n', line, j)

			val = val | 1 << (j - 4)

	return val

#
# Takes a string specifying the support qualifer and returns
# the corresponding code
#
def gen_qual(str):
	ret=0
	for i in range(0, len(str)):
		if str[i] not in qual_list:
			error('line %d: unknown qualifier %c\n', line, str[i])
		#
		# special treatment of the '-'
		# if '-' present, it can be the only one specified
		#
		if str[i] == '-':
			if i>0 or len(str) != 1 : error('line %d: the - qualifier cannot be used with others\n', line)
			return 0

		ret = ret | 1 << i

	return ret	

#
# main function
#	
if len(sys.argv) == 1: 
	error('You need to specify the database file !\n')

f = open(sys.argv[1], 'r')

line=-1
i,entry_id=0,0

#
# verbatim output of the C array
#
print """
/* Copyright (C) 2001 Hewlett-Packard Co
 * Copyright (C) 2001 Stephane Eranian <eranian@hpl.hp.com>
 */

/*
 * This part of the code is generated automatically
 * !! DO NOT CHANGE !!
 * Make changes to the database file perfmon.list instead
 */
static pme_entry_t pe[]={
"""
#
# read lines and process them
#
for l in f.readlines():

	#
	# current line in file
	#
	line=line+1

	# skip comment lines
	if l[0]=='#': continue

	# transform into list of elements
	p = string.split(l[:-1], '\t')

	# filter out '' elements
	p = filter(empty,p)

	# takes care of empty lines
	if len(p)==0: continue;

	#
	# umask cannot have more that four digits
	if len(p[2]) != 4: error('line %d: Invalid umask %s\n', line, p[2])

	umask=gen_umask(p[2])

	print '#define PME_%s %d' % (string.replace(p[0],'.','_'), entry_id)

	#
	# extract counter constraints
	#
	cnt = gen_counter(string.split(p[4],','))

	qual=gen_qual(p[5])	

	# we ignore any trailing parameters

	code = (string.atoi(p[1],0) & 0xffff) | (umask &0xffff) << 16
	print '{ "%s", {0x%x} , %d, 0x%x, {0x%x}},' % (p[0], code, string.atoi(p[3],0), cnt, qual)

	#
	# entry counter in table
	#
	entry_id=entry_id+1

f.close()

print """{ NULL, {0}, 0,0, {0} }};"""

print "#define PME_COUNT", entry_id



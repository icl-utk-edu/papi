#!/usr/bin/python
#
# genpme.py: generate event lists from the event database
#
# Copyright (C) 2001-2002 Hewlett-Packard Co
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

import sys,string,os

#
# global parameters
#

#
# Our own exception
#
event_error = 'EventError'

# list of implements counters(Itanium has only 4)
# use tuples for immutability
impl_counters=[]

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
#	sys.stderr.write(fmt % args)
	msg=fmt % args
	raise event_error, msg
#
# file printing routine
#
def fprintf(file, fmt, *args):
	file.write(fmt % args)


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
			error('invalid umask 0x%x\n', p[i])

		val = val | val_lst.index(p[i])*pow(2,(3-i))
	return val

#
# Convert list of implemented counters from
# string elements to integer elements to allow the use
# of in operator for the test
#
def convert_counters(p):
	for i in range(0, len(p)):
		impl_counters[:0]=[ string.atoi(p[i]) ]
	
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
				error('invalid counter %d\n',j)
			val = val | 1 << (j - 4)

	return val << 4

#
# Takes a string specifying the support qualifer and returns
# the corresponding code
#
# qual structure:
#	bit 0      : I
#	bit 1      : O
# 	bit 2      : D
#	bit 3 	   : reserved
#	bit [4:15] : reserved
#	bit [16:19]: group (default=0xf)
#	bit [20:23]: set (default=0xf)
#	bit [24:31]: setctr (default=0xff)
#
def gen_qual(q, grp_str, set_str, setctr_str):
	qual = 0
	if len(q) > 3 : error('invalid qualifier %s\n', q)
	for i in range(0, len(q)):
		if q[i] not in qual_list:
			error('unknown qualifier %c\n', q[i])
		#
		# special treatment of the '-'
		# if '-' present, it can be the only one specified
		#
		if q[i] == '-':
			if i>0 or len(q) != 1 : error('the - qualifier cannot be used with others\n')
			qual = 0
		else:
			qual = qual | 1 << i
	
	setctr = string.atoi(setctr_str);
	grp    = string.atoi(grp_str);
	set    = string.atoi(set_str);
	return (setctr & 0xff) << 24 | (set & 0xf) << 20 | (grp & 0xf)<< 16 | qual
#
# main function
#	
if len(sys.argv) == 1: 
	error('You need to specify the database file !\n')

if len(sys.argv) == 3:
	try:
		outfile=open(sys.argv[2], "w")
	except IOError:
		print 'Cannot create file', sys.argv[2]
		sys.exit(1)
	
else:
	outfile=sys.stdout

line=0
i,entry_id=0,0

# setup default names
vnames=['Generic','GEN','pme_entry_t','generic_pe']

try:
	f = open(sys.argv[1], 'r')

	#
	# verbatim output of the C array
	#
	fprintf(outfile, """
/* Copyright (C) 2001-2002 Hewlett-Packard Co
 *               Stephane Eranian <eranian@hpl.hp.com>
 */

/*
 * This part of the code is generated automatically
 * !! DO NOT CHANGE !!
 * Make changes to the database file instead
 */
""")

	#
	# read lines and process them
	#
	for l in f.readlines():

		#
		# current line in file
		#
		line=line+1

		# remove leading white spaces
		l = string.lstrip(l)

		if len(l) == 0: continue

		# skip comment lines
		if l[0]=='#': continue

		# extract model information
		if len(l) >= 6 and l[:6]=='model=' : 
			vnames = string.split(l[6:-1],',');
			if len(vnames) != 4: error("invalid model description");
			continue

		# extract counter information
		if len(l) >= 9 and l[:9]=='counters=' : 
			convert_counters(string.split(l[9:-1],','));
			continue

		# convert to known separator if needed
		l = string.replace(l, ' ', '\t')

		# transform into list of elements
		p = string.split(l[:-1], '\t')

		# filter out '' elements
		p = filter(empty,p)

		# takes care of empty lines
		if len(p)==0: continue;
		#
		# umask cannot have more that four digits
		#if len(p[2]) != 4: error('Invalid umask %s\n', p[2])

		#umask=gen_umask(p[2])
		umask=string.atoi(p[2], 16)

		if entry_id == 0 :
			fprintf(outfile, "/*\n * Events table for %s processor\n */\n", vnames[0])
			fprintf(outfile, "static %s %s[]={\n", vnames[2], vnames[3]);

		fprintf(outfile, "#define PME_%s_%s %d\n", vnames[1], string.replace(p[0],'.','_'), entry_id);

		#
		# extract counter constraints
		#
		cnt = gen_counter(string.split(p[4],','))

		qual=gen_qual(p[5], p[6], p[7], p[8])	

		# we ignore any trailing parameters


		code = (string.atoi(p[1], 0) & 0xffff) | (umask &0xffff) << 16
		fprintf(outfile,'{ "%s", {0x%x} , 0x%x, %d, {0x%x}},\n', p[0], code, cnt, string.atoi(p[3],0), qual);

		#
		# entry counter in table
		#
		entry_id=entry_id+1

	f.close()

	fprintf(outfile, """{NULL, {0}, 0, 0, {0}}};\n""");

	fprintf(outfile, "#define PME_%s_COUNT %d\n", vnames[1], entry_id);

except IOError:
	print 'Cannot read from', sys.argv[1]
	outfile.close()
	sys.exit(1)

except event_error, msg: 
	print 'Error line', line, 'in', sys.argv[1], ':',msg
	outfile.close()
	if len(sys.argv) == 3: os.unlink(sys.argv[2])
	sys.exit(1)



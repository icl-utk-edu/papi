--------------------------ADDING PCP EVENTS TO PAPI----------------------------

This README is intended for developers that understand the Linux command line
well enough to execute a shell script, 'make' a file, 'mv' files to rename
them or to relocate them to another directory, and use an editor to change
characters in a .txt file. They will also need to rebuild the PAPI library,
and execute a script or command. If this requires 'srun' or 'jsrun' or some
other batch script language on a cluster, they need to know how to do that.

The PCP component will always add all PCP events that begin with 'perfevent'.

Other events can be provided in an array of strings; this is defined in a file
called PCPEventList.c, that resides in the same directory as the component
code itself; linux-pcp.c. This directory is papi/src/components/pcp/ and it is
the parent directory of where this README is found.

PCPEventList.c is #include'd in linux-pcp.c; and it must reside in the same
directory for the PAPI build script to find it when it compiles linux-pcp.c. 
 
However, we do not recommend changing PCPEventList.c directly. It is a
generated file; based on the contents of pmid.txt ('pm' is for Performance
Monitor; a part of PCP). 

The program that uses pmid.txt to write out the file PCPEventList.c is called
buildPCPEventList. This is a simple program, it constructs a string array for
C-code that contains only the events on lines of pmid.txt that do not begin
with the character '#'. Any line beginning with '#' is ignored.

pmid.txt contains all the PCP events (if it does not or is out of date, we
will show how to build a new one below). So using an editor, find the events
you wish to include, and change the first character to a space. (Don't change
it to anything else, it should be a space, or '#'). 

Then you can compile buildPCPEventList like so:

> make -f Makefile2 buildPCPEventList

and run it: 

> ./buildPCPEventList

and it will output PCPEventList.c. 

Note this will be in the /papi/src/components/pcp/tests/ directory. You should
check PCPEventList.c to verify it is the expected code. If you wish to backup
the previous version (for safety) in the parent directory, rename it:

> mv ../PCPEventList.c ../orig-PCPEventList.c 

Then move this version to the parent directory:

> mv PCPEventList.c ../.


-----------------------------REBUILDING pmid.txt-------------------------------

The primary reason to do this is if your pmid.txt does not represent all of
the events available to whatever version of PCP you have installed. PCP comes
with a utility, 'pminfo', which can list all the events. If your work machine
accesses PCP directly, then 

> pminfo -t 

will list all the events available to you. If your machine is just used to
schedule a job to be run on some node, for example via via srun or jsrun, then
you need to run pminfo using that:

> jsrun --np 1 pminfo -t

It might be useful to redirect that output to a file; there can be several
thousand such events:

> jsrun --np 1 pminfo -t >capture-pminfo.txt

We provide a script, newpmid.sh, that will use pminfo to extract a new list of
events available to the machine in question, and change it to the format of
the pmid.txt we use with buildPCPEventList.c. It will capture the output, sort
it, add comments to the top and prefix every line of the pminfo output with a
'# '. The result is in 'newpmid.txt'.

This script can be executed as

> sh newpmid.sh "any prefix needed to run a program"

For example, on my test cluster, I required: 

> sh newmpid.sh "jsrun --np 1"

The '#' character comments out all of the events, so NONE of them will be 
added to PAPI.

You can edit newpmid.txt, change the first character of a line from '#' to
space will enable the PCP events you want. You could then rename newpmid.txt
to pmid.txt, and run buildPCPEventList to create a new PCPEventList.c file. 
See the first topic at top of this README for more detailed instructions.


HOWEVER, there is another program that will copy all the enables of pmid.txt
to the newpmid.txt, if the events are found in newpmid.txt. It is called 
updatePMID.c. 

To run this, first rename pmid.txt to oldpmid.txt:

> mv pmid.txt oldpmid.txt

Compile updatePMID.c:

> make -f Makefile2 updatePMID

And execute it. IT WILL REPLACE ANY pmid.txt FILE, if you want what you had
for reference make sure you made a copy or did the rename above. To execute:

> ./updatePMID

It will read all the enabled events in oldpmid.txt, find them in newpmid.txt,
and change the first character from '#' to space before writing it into
pmid.txt.  Any enabled events in oldpmid.txt that are NOT found in newpmid.txt
are noted on stderr. It leaves unchanged oldpmid.txt and newpmid.txt; the
changes are only present in the output file, pmid.txt. No comments you have
added to the previous pmid.txt are preserved, either; if you are using
comments to explain the utility of some vents or anything else, you must
transfer those manually. (I would suggest if you wish to keep notes in
pmid.txt, then start your own comments with '##' so you can find them easily.)

Then you can edit pmid.txt and enable any OTHER events you might like to see,
as per the instructions on this at the top of this README.

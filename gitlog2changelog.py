#!/usr/bin/python
# Copyright 2008 Marcus D. Hanwell <marcus@cryos.org>
# Distributed under the terms of the GNU General Public License v2 or later

import string, re, os
import sys 

# Execute git log with the desired command line options.
command="git log --summary --stat --no-merges --date=short "
if len(sys.argv) < 2:
	print "Please specify a tag."
	sys.exit(1)

command += sys.argv[1] + ".." 
fin = os.popen(command, 'r')
# Create a ChangeLog file in the current directory.
fout = open('ChangeLog', 'w')

# Set up the loop variables in order to locate the blocks we want
authorFound = False
dateFound = False
messageFound = False
filesFound = False
message = ""
messageNL = False
files = ""
prevAuthorLine = ""
commitID = ""

# The main part of the loop
for line in fin:
    # The commit line marks the start of a new commit object.
    if string.find(line, 'commit') >= 0:
        # Start all over again...
        authorFound = False
        dateFound = False
        messageFound = False
        messageNL = False
        message = ""
        filesFound = False
        files = ""
        foo = line.split( None )
        commitID = foo[1][:8]
        continue
    # Match the author line and extract the part we want
    elif re.match('Author:', line) >=0:
        authorList = re.split(': ', line, 1)
        author = authorList[1]
        author = author[0:len(author)-1]
        author = ""
        authorFound = True
    # Match the date line
    elif re.match('Date:', line) >= 0:
        dateList = re.split(':   ', line, 1)
        date = dateList[1]
        date = date[0:len(date)-1]
        dateFound = True
    # The svn-id lines are ignored
    elif re.match('    git-svn-id:', line) >= 0:
        continue
    # The sign off line is ignored too
    elif re.search('Signed-off-by', line) >= 0:
        continue
    # Extract the actual commit message for this commit
    elif authorFound & dateFound & messageFound == False:
        # Find the commit message if we can
        if len(line) == 1:
            if messageNL:
                messageFound = True
            else:
                messageNL = True
        elif len(line) == 4:
            messageFound = True
        else:
            if len(message) == 0:
                message = message + line.strip()
            else:
                message = message + " " + line.strip()
    # If this line is hit all of the files have been stored for this commit
    elif any(x in line for x in ['files changed', 'file changed']):
        filesFound = True
        continue
    # Collect the files for this commit. FIXME: Still need to add +/- to files
    elif authorFound & dateFound & messageFound:
        fileList = re.split(' \| ', line, 2)
        if len(fileList) > 1:
            if len(files) > 0:
                files = files + ", " + fileList[0].strip()
            else:
                files = fileList[0].strip()
    # All of the parts of the commit have been found - write out the entry
    if authorFound & dateFound & messageFound & filesFound:
        # First the author line, only outputted if it is the first for that
        # author on this day
        authorLine = date + "  " + author
        if len(prevAuthorLine) == 0:
            fout.write(authorLine + "\n")
        elif authorLine == prevAuthorLine:
            pass
        else:
            fout.write("\n" + authorLine + "\n")

        pruned_files = files.split(",")
        files = ""
        for i in range( len(pruned_files) )[:3]:
            files+= pruned_files[i]
        if len(pruned_files) > 3:
            files+="..."
# Assemble the actual commit message line(s) and limit the line length
# to 80 characters.
        commitLine = "* " + commitID + " " + files + ": " + message
        i = 0
        commit = ""
        while i < len(commitLine):
            if len(commitLine) < i + 78:
                commit = commit + "\n  " + commitLine[i:len(commitLine)]
                break
            index = commitLine.rfind(' ', i, i+78)
            if index > i:
                commit = commit + "\n  " + commitLine[i:index]
                i = index+1
            else:
                commit = commit + "\n  " + commitLine[i:78]
                i = i+79

        # Write out the commit line
        fout.write(commit + "\n")

        #Now reset all the variables ready for a new commit block.
        authorFound = False
        dateFound = False
        messageFound = False
        messageNL = False
        message = ""
        filesFound = False
        files = ""
        prevAuthorLine = authorLine

# Close the input and output lines now that we are finished.
fin.close()
fout.close()

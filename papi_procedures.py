#!/usr/bin/env python3
import os, re, sys, string, argparse, getpass
from textwrap import TextWrapper

'''
This Python script serves as a source of documentation and as an automated process for a new PAPI release.
'''
def verify_user_input(prompt: str):
    '''Helper function to verify either y or n are provided on the command line.

    Args:
        prompt (str): A set of instructions that will be output to the screen.
    '''
    userInput = None
    while userInput != "y" and userInput != "n":
        userInput = input(f"{prompt}")

    return userInput

def read_and_write_papi_version(filename: str, varNameToSearch: str, varNameToReplace: str):
    '''Helper function to read and write the new PAPI version to filename.

    Args:
        filename (str): Name of the file that needs a version update.
        varNameToSearch (str): Variable that will be searched for to help update the version number.
        varNameToReplace (str): The variable with the new updated PAPI version.
    '''
    with open(filename, "r") as file:
        contentsOfFile = file.read()
        updatedContentsOfFile = re.sub(f"{varNameToSearch}.*", varNameToReplace, contentsOfFile)

    with open(filename, "w") as file:
        file.write(updatedContentsOfFile)

def update_papi_version_in_files(newVersionNumber: str):
    '''Helper function to update the version numbers in papi.spec, src/papi.h,
       src/configure.in, doc/Doxyfile-common, and src/Makefile.in.

    Args:
        newVersionNumber (str): The new PAPI version number in X.Y.Z.N format (e.g. 7.2.0.0).
    '''
    major, minor, revision, increment = newVersionNumber.split(".") 
    filenames = ["papi.spec", "src/papi.h", "src/configure.in", "doc/Doxyfile-common"]
    varNamesToSearch = ["Version:", "#define PAPI_VERSION  ", "AC_INIT", "PROJECT_NUMBER  "]
    varNamesToReplace = [f"Version: {newVersionNumber}", f"#define PAPI_VERSION  \t\t\tPAPI_VERSION_NUMBER({major},{minor},{revision},{increment})",
                         f"AC_INIT(PAPI, {newVersionNumber}, ptools-perfapi@icl.utk.edu)", f"PROJECT_NUMBER         = {newVersionNumber}"] 
    for filename, varNameToSearch, varNameToReplace in zip(filenames, varNamesToSearch, varNamesToReplace):
        read_and_write_papi_version(filename, varNameToSearch, varNameToReplace)

    # Update the version numbers in src/Makefile.in
    for varname, ver in zip(["PAPIVER", "PAPIREV", "PAPIAGE", "PAPIINC"], [major, minor, revision, increment]):
        read_and_write_papi_version("src/Makefile.in", varname, f"{varname}={ver}")

def generate_change_log(startingCommit: str, outputFilename: str):
    '''Generate a change log based off the commits since the previous release.

    Args:
        startingCommit (str): The commit hash which determines the starting point
                              for the commits listed in the change log. Note that
                              this starting commit is NON-INCLUSIVE.
        outputFileName (str): The name of the change log where you want the commits
                              to be stored. E.g. ChangeLogP720.txt.
    '''
    # Range of specific commits that we want to create a change log for
    rev_range = '%s..HEAD' % startingCommit

    # Execute git log with the desired command line options.
    # This is implemented using subprocess.Popen
    fin = os.popen('git log --summary --stat --no-merges --date=short %s' % rev_range, 'r', buffering = -1)
    # Needed to properly parse final entry 
    lines = fin.readlines()
    last_line = lines[-1]
    # Create a ChangeLog file in the current directory.
    fout = open(outputFilename, 'w')

    # Set up the loop variables in order to locate the blocks we want
    authorFound = False
    dateFound = False
    messageFound = False
    filesFound = False
    message = ""
    messageNL = False
    files = ""
    prevAuthorLine = ""

    wrapper = TextWrapper(initial_indent="\t", subsequent_indent="\t  ")

    # The main part of the loop
    for line in lines:
        # The commit line marks the start of a new commit object.
        if line.startswith('commit'):
            # Start all over again...
            authorFound = False
            dateFound = False
            messageFound = False
            messageNL = False
            message = ""
            filesFound = False
            files = ""
            continue
        # Match the author line and extract the part we want
        elif 'Author:' in line:
            authorList = re.split(': ', line, 1)
            author = authorList[1]
            author = author[0:len(author)-1]
            authorFound = True
        # Match the date line
        elif 'Date:' in line:
            dateList = re.split(':   ', line, 1)
            date = dateList[1]
            date = date[0:len(date)-1]
            dateFound = True
        # The Fossil-IDs are ignored:
        elif line.startswith('    Fossil-ID:') or line.startswith('    [[SVN:'):
            continue
        # The svn-id lines are ignored
        elif '    git-svn-id:' in line:
            continue
        # The sign off line is ignored too
        elif 'Signed-off-by' in line:
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
        elif re.search('files? changed', line) != None:
            filesFound = True
            # We only want to continue if it is not the last line;
            # continuing on the last line would skip the final entry
            if line is not last_line:
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
                fout.write(authorLine + "\n\n")
            elif authorLine == prevAuthorLine:
                pass
            else:
                fout.write("\n" + authorLine + "\n\n")

            # Assemble the actual commit message line(s) and limit the line length
            # to 80 characters.
            commitLine = "* " + files + ": " + message

            # Write out the commit line
            fout.write(wrapper.fill(commitLine) + "\n")

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

def papi_release():
    '''Goes through the PAPI release procedures.'''
    # Spaces used for formatting output, currently there are 7 total spaces to help offset the indention of "Step #: ".
    spacesForFormatting = "       "

    '''
    Update the man pages, configure, and the RELEASENOTES along with creating a ChangeLog 
    '''
    # Step 0, this step must be completed before continuing. Without this being done the rest of the release
    # script will not work properly.
    promptForUser = ("Step 0: Before proceeding, you must have a fork of the PAPI repository (https://github.com/icl-utk-edu/papi) and you must be in the root\n"
                     f"{spacesForFormatting} directory of this fork (cd papi/) on the master branch with the most upto date commits. Are these requirements met (y or n)? ") 
    preemptiveStep = verify_user_input(promptForUser)
    if preemptiveStep != "y":
        print("\033[31mStep 0 must be completed before continuing.\033[0m")
        sys.exit(1)

    print(" ")
    # Step 0.1, provide the name of the remote repository that coincides with your fork.
    # Must be provided to properly run `git push` later.
    forkRemoteName = input("Step 0.1: Provide the name of the remote repository that coincides with your fork (oftentimes this is called `origin`): ")

    print(" ")
    # Step 1, create a new branch. The branch name can really be anything, but common naming schemes are papi-release-#-#-# or papi-release-#-#-#-t.
    branchName = input("Step 1: A new branch needs to be created, provide a name (e.g. papi-release-7.2.0): ")
    os.system(f"git checkout -b {branchName}")

    print(" ")
    # Step 2, optionally update the documentation namely INSTALL.txt IF needed. On most occassions
    # this will already have been completed with prior commits that have already been merged into master.
    promptForUser = ("Step 2: Does the documentation need to be updated (e.g. INSTALL.txt) (y or n)? ")
    updateDocs = verify_user_input(promptForUser)
    if updateDocs == "y":
        input(f"{spacesForFormatting} Documentation needs to be updated. Press ctrl + z to suspend this script. Once the documentation has been updated, on the command line run `jobs`\n"
              f"{spacesForFormatting} locate papi_procedures.py, and finally run `fg %#` where # is the value in brackets in the left most column. Once papi_procedures.py is brought to the\n"
              f"{spacesForFormatting} foreground then Press Enter to Continue.")

    print(" ")
    # Step 3, check/change the version number in papi.spec, src/papi.h, src/configure.in, src/Makefile.in, doc/Doxyfile-common.
    promptForUser = ("Step 3: Does the version number need to be updated (y or n)? ")
    updateVersions = verify_user_input(promptForUser)
    if updateVersions == "y":
        updatedVersionNumber = input(f"{spacesForFormatting} Provide the updated version number (include major, minor, revision, and increment e.g. 7.2.0.0): ")

        update_papi_version_in_files(updatedVersionNumber)

        ## Add and commit these updated files.
        input(f"{spacesForFormatting} The files papi.spec, src/papi.h, src/configure.in, src/Makefile.in, and doc/Doxyfile are being added, please add a commit message. (Press Enter to Continue).")
        os.system("git add papi.spec; git add src/papi.h; git add src/configure.in; git add src/Makefile.in; git add doc/Doxyfile-common; git commit")

    print(" ")
    # Step 4, rebuilding the doxygen manpages.
    input("Step 4: Rebuilding the doxygen manpages. (Press Enter to Continue).")
    os.system("module load doxygen; cd doc; make; make install; git status")
    input("\033[93m`git status` was just ran as Doxygen could generate extraneous files such as man/man1/_home_youruserid_*.* and man/man3/_home_youruserid_*.*.\n"
          "Verify this is not the case and suspend the application if needed to remove extraneous files. (Press Enter to Continue).\033[0m")
    os.system("cd man/man1; git add *")
    os.system("cd man/man3; git add *")

    print(" ")
    # Step 5, rebuilding the website docs.
    # As of June 25th, 2025 the web dir is located on icl.utk.edu with the path of websites/icl.utk.edu/projectsdev/papi/docs.
    # Due to this, it is recommended to open up another terminal window and then ssh to icl.utk.edu. However, you can either ssh directly
    # to icl.utk.edu from your local machine or you can be on the login node at ICL (login.icl.utk.edu).
    # Note that you will have to authenticate using your SSH key.
    # The files will need to have group write permissions for certain operations. The "docs" directory does have group
    # write permissions; therefore, overwriting the files there should be possible, but it may not be possible to open the files and
    # edit them. Contact the sysadmin if needed which currently is Geri Ragghianti.
    input("Step 5: Rebuilding the website docs. (Press Enter to Continue).")
    input("Step 5.1: Open up another terminal window and ssh to icl.utk.edu. Note that you must authenticate with your ssh key. (Press Enter to Continue).")
    input("Step 5.2: Remove the current docs (/bin/rm -rf websites/icl.utk.edu/projectsdev/papi/docs/*). (Press Enter to Continue).")
    input("Step 5.3: The docs will now be rebuilt automatically (Press Enter to Continue).")
    os.system("cd doc; make clean html")
    input("Step 5.4: The docs should have been successfully rebuilt. If they have not suspend the script. (Press Enter to Continue).")
    input("Step 5.5: The newly rebuilt docs will be `scp'd` to websites/icl.utk.edu/projectsdev/papi/docs. (Press Enter to Continue).")
    username = getpass.getuser()
    os.system(f"scp -r doc/html {username}@icl.utk.edu:/home/{username}/websites/icl.utk.edu/projectsdev/papi/docs")
    input("Step 5.6: On the second terminal window change the permissions to 775 on websites/icl.utk.edu/projectsdev/papi/docs (e.g. chmod -R 775 websites/icl.utk.edu/projectsdev/papi/docs). (Press Enter to Continue).")
    input("Step 5.7: Verify that the website docs have been replaced and the permissions changed to 775. (Press Enter to Continue).")

    print(" ")
    # Step 6, if configure.in has changed, we must generate a new `configure` file. We want to use autoconf version 2.69, currently
    # this version of autoconf can be loaded on methane at ICL with `module load autoconf-archive`.
    promptForUser = ("Step 6: Does configure need to be updated (y or n)? ")
    updateConfigure = verify_user_input(promptForUser)
    if updateConfigure == "y":
        os.system("""module load autoconf-archive;
                     cd src;
                     autoconf configure.in > configure;
                     git add configure""")

    print(" ")
    # Step 7, create a ChangeLog for the current release.
    print("Step 7: A ChangeLog will now be generated. (Press Enter to Continue).")
    startingCommit = input(f"{spacesForFormatting} Enter a starting commit (non-inclusive): ")
    nameOfChangeLog = input(f"{spacesForFormatting} Enter a name for the change log (e.g. ChangeLogP720.txt): ")
    generate_change_log(startingCommit, nameOfChangeLog)

    print(" ")
    # Step 8, scan the ChangeLog to remove extraneous fluff, like perfctr imports.
    input(f"Step 8: {nameOfChangeLog} will now be opened, scan the file to remove any extraneous fluff. (Press Enter to Continue).")
    os.system(f"vi {nameOfChangeLog}")

    print(" ")
    # Step 9, modify the RELEASENOTES.txt to summarize the major changes listed in the ChangeLog.
    input("Step 9: RELEASENOTES.txt will now be opened, summarize the major changes for this release. (Press Enter to Continue).")
    os.system("vi RELEASENOTES.txt")

    print(" ")
    # Step 10, add and commit ChangeLogPXYZ.txt and RELEASENOTES.txt. Along with the two aforementioned files configure will also be commited at
    # this stage.. 
    input(f"Step 10: {nameOfChangeLog} and RELEASENOTES.txt will now be added. Please add a commit message for {nameOfChangeLog}, RELEASENOTES.txt, and configure (added at Step 6). (Press Enter to Continue).")
    os.system(f"git add {nameOfChangeLog}; git add RELEASENOTES.txt; git commit")

    print(" ")
    # Step 11, push the branch to the remote fork.
    input(f"Step 11: Commited changes will now be pushed to the remote fork. (Press Enter to Continue).")
    os.system(f"git push {forkRemoteName} {branchName}")

    print(" ")
    # Step 12, Create a pull request. Having just made a push, a yellow banner should appear right above the green "Code" dropdown button.
    # If this is not the case then follow the steps outlined below:
    # 1. Click on the "Pull requests" button located in the top left corner.
    # 2. Click on the green "New pull request" button located in the middle right corner.
    # 3. Click on the blue "compare across forks" button located below the "Compare changes" heading.
    # 4. For the head repository select your PAPI fork and then for compare select the branch that you would like to merge in.
    # 5. Review the changes and if everything looks correct click "Create pull request".
    input(f"Step 12: Go to https://github.com/icl-utk-edu/papi and create a pull request for the branch {branchName}. Wait for this pull request to be reviewed, approved, and merged. (Press Enter to Continue).")

    '''
    Branching and Tagging
    '''
    print(" ")
    input("With Step 12 completed we will now move onto branching and tagging. (Press Enter to Continue).")

    print(" ")
    # Step 13, cloning the new PAPI.
    input("Step 13: Clone the new PAPI. To do this without leaving the current directory, we will clone PAPI with a new directory name titled 'tag_papi'. (Press Enter to Continue).")
    os.system("git clone https://github.com/icl-utk-edu/papi.git tag_papi")

    print(" ")
    # Step 14, branch PAPI if it is not an incremental release.
    promptForUser = ("Step 14: Is this an incremental release (y or n)? ")
    incrementalRelease = verify_user_input(promptForUser) 
    nameOfTagBranch = None
    if incrementalRelease == "n":
        nameOfTagBranch = input(f"{spacesForFormatting}  Not an incremental release, please provide a branch name to branch git (e.g. stable-7.2.0): ")
        os.system(f"cd tag_papi; git checkout -b {nameOfTagBranch}")
    else:
        input("""The original release_procedure.txt that this Python script adopts from did not have incremental release directions.
The rest of the script assumes this is not an incremental release. If this has changed exit and update this script.""")

    print(" ")
    # Step 15, tag PAPI and push the tag to the central repo. You will be prompted for a comment on the tags. A tags comment should be able to be seen
    # by clicking "Tags" on GitHub and then clicking the desired tag you would like to see. For a comment, "Release PAPI-7-2-0-t" is sufficient (your own version of course).
    nameOfTag = input("Step 15: A tag will now be created, please provide a name for the tag (e.g. papi-7-2-0-t, note that for this tag you will be prompted for a comment): ")
    os.system(f"cd tag_papi; git tag -a {nameOfTag}")
    if incrementalRelease == "n":
        os.system(f"cd tag_papi; git push --tags origin {nameOfTagBranch}")
    else:
        os.system(f"cd tag_papi; git push --tags")

    print(" ")
    # Step 16, verify that the branch and tag appears in the repo.
    input("Step 16: Go to https://github.com/icl-utk-edu/papi and verify that both the branch and tag have been created. (Press Enter to Continue).")

    '''
    Build a Tarball
    '''
    print(" ")
    input("Now that branching and tagging has been completed a tarball will now be created. (Press Enter to Continue).")

    print(" ")
    # Step 17, clone PAPI again but under a directory name including the release number.
    # This is important as we will delete the .git file for the tarball, so this new
    # directory will no longer be under git.
    nameOfTarballDir = input("Step 17: Provide a name to clone PAPI under (e.g. papi-7.2.0, note that it must have the release number included): ")
    os.system(f"git clone https://github.com/icl-utk-edu/papi.git {nameOfTarballDir}")

    print(" ")
    # Step 18, deleting the unneccessary files or directories particularly .doc and .pdf
    # files in the /doc directory.
    input("Step 18: Deleting unneccessary files/directories. (Press Enter to Continue).")
    os.system(f"""cd {nameOfTarballDir};
                  rm PAPI_FAQ.html;
                  rm doc/DataRange.html;
                  rm doc/PAPI-C.html;
                  rm doc/README;
                  rm src/buildbot_configure_with_components.sh;
                  rm papi_procedures.py;
                  rm -rf .git;
                  rm -rf .github;
                  rm .gitattributes""")

    print(" ")
    # Step 19, the directory name provided in Step 17 will now be tar'd, zip'd, and scp'd to the website.
    pathToDownloads = f"/home/{username}/websites/icl.utk.edu/projects/papi/downloads/" 
    input(f"Step 19: The directory {nameOfTarballDir} will now be tar'd, zip'd, and scp'd to {pathToDownloads}. (Press Enter to Continue).")
    os.system(f"""tar -cvf {nameOfTarballDir}.tar {nameOfTarballDir};
                  gzip {nameOfTarballDir}.tar;
                  chmod 664 {nameOfTarballDir}.tar.gz;
                  scp {nameOfTarballDir}.tar.gz {username}@icl.utk.edu:{pathToDownloads}""")

    print(" ")
    # Step 20, verify that you can download the tarball. The landing page may look broken, but as long as the
    # tarball is downloaded from your created link, then everything is working as expected.
    input(f"Step 20: Copy and paste the following link http://icl.utk.edu/projects/papi/downloads/{nameOfTarballDir}.tar.gz to verify the created tarball is downloaded. (Press Enter to Continue).")

    print(" ")
    # Step 21, create a link with supporting text on the PAPI software web page and on the PAPI wiki.
    input("Step 21: Create a link with supporting text on the PAPI software web page and on the PAPI wiki (https://github.com/icl-utk-edu/papi/wiki/PAPI-Releases). (Press Enter to Continue).")

    print(" ")
    # Step 22, create a news item on the PAPI web page.
    input("Step 22: Create a news item on the PAPI web page. (Press Enter to Continue).")

    print(" ")
    # Step 23, email the PAPI developer and discussion lists with an announcement in regards to the release.
    input("Step 23: Email the PAPI developer (perfapi-devel@icl.utk.edu) and discussion lists (ptools-perfapi@icl.utk.edu) announcing the release. (Press Enter to Continue).")

    '''
    Bump Version Number 
    '''
    print(" ")
    input("With Step 23 completed the version number will now be bumped. (Press Enter to Continue).")

    print(" ")
    # Step 23, get the most recent changes to the master branch so that we can bump the version number properly.
    input("Step 23: The master branch will now be checked out and the most recent changes will be fetched. (Press Enter to Continue).")
    os.system("git checkout master; git fetch upstream master && git reset --hard FETCH_HEAD")

    print(" ")
    # Step 24, checkout a new branch to bump the version numbers.
    nameOfBumpBranch = input("Step 24: Provide a new branch name to checkout, this branch will be used to bump the version numbers: ")
    os.system(f"git checkout -b {nameOfBumpBranch}")

    print(" ")
    # Step 25, bump the version numbers in the repository AFTER the release changing Z to Z + 1 (e.g. 7.2.0 (X.Y.Z) to 7.2.1)
    bumpedVersionNumber = input("Step 25: Provide the bumped version number to update papi.spec, src/papi.h, src/configure.in, src/configure, src/Makefile,\n"
                                f"{spacesForFormatting}  and doc/Doxyfile-common. As an example if your release was 7.2.0.0 then bump the version number to 7.2.1.0: ")
    update_papi_version_in_files(bumpedVersionNumber)

    print(" ")
    # Step 26, generate a new configure.
    input("Step 26: A new configure will now be generated. (Press Enter to Continue).")
    os.system("cd src; autoconf configure.in > configure")

    print(" ")
    # Step 27, add and commit the necessary files that have been changed during the bump version process.
    input("Step 27: The files papi.spec, src/papi.h, src/configure.in, src/configure, src/Makefile.in, and doc/Doxyfile-common are being added and will be pushed, please add a commit message. (Press Enter to Continue).")
    os.system(f"""git add papi.spec src/papi.h src/configure.in src/configure src/Makefile.in doc/Doxyfile-common;
                  git commit;
                  git push {forkRemoteName} {nameOfBumpBranch}""")

    print(" ")
    # Step 28, create a pull request for the files that were changed during the bump version process.
    input(f"Step 28: Go to https://github.com/icl-utk-edu/papi and create a pull request for the branch {nameOfBumpBranch}. (Press Enter to Continue).")

    print(" ")
    # Step 29, if the papi_procedures.py script needs to be updated do so.
    input("Step 29: If during the release process the steps in this script (papi_procedures.py) resulted in incorrect behavior create a new branch and fix them. (Press Enter to Continue).")

    print("\033[32mThe release procedures have been completed!\033[0m")

def papi_bugfix():
    '''Goes through the PAPI bug fix procedures.'''
    spacesForFormatting = "       "
    # Step 0, clone PAPI.
    promptForUser = ("Step 0: Before proceeding, are you in a fresh clone of PAPI (git clone https://github.com/icl-utk-edu/papi.git) and in the root dir (cd papi/) (y or n)? ")
    preemptiveStep = verify_user_input(promptForUser)
    if preemptiveStep != "y":
        print("\033[31mStep 0 must be completed before continuing.\033[0m")
        sys.exit(1)

    print(" ")
    # Step 1, determine if this is the first bug fix release.
    # If it is the first bug fix release we simply can checkout the current release branch.
    # If it is not the first bug fix release we will need to create a branch based off the last bug fix release tag.
    promptForUser = ("Step 1: Is this the first bug fix (y or n)? ")
    firstBugFix = verify_user_input(promptForUser)
    bugFixBranchName = None
    if firstBugFix == "y":
        bugFixBranchName = input(f"{spacesForFormatting} Provide the name of the release branch you are wanting to apply a bug fix for (e.g. stable-7.2.0): ")
        os.system(f"git checkout {bugFixBranchName}")
    elif firstBugFix == "n":
        input(f"{spacesForFormatting} As this is not the first bug fix for this release we must create a branch from the last bug fix release. (Press Enter to Continue).")
        previousTag = input(f"{spacesForFormatting} Provide the tag of the previous bug fix release (e.g. tags/papi-7-2-0-1-t): ")
        bugFixBranchName = input(f"{spacesForFormatting} Provide a branch name (e.g. papi-7-2-0-2, notice the increment in the incremental): ")
        os.system(f"git checkout {previousTag} -b {bugFixBranchName}")

    print(" ")
    # Step 2, verify that we are on the branch we want to apply bug fixes for.
    os.system("git branch")
    promptForUser = ("Step 2: `git branch` was just ran. Are you on the correct branch to apply the bug fixes (y or n)? ")
    correctBranch = verify_user_input(promptForUser)
    if correctBranch == "n":
         print("\033[31mOn incorrect branch, exiting the papi_procedures script.\033[0m")
         sys.exit(1) 

    print(" ")
    # Step 3, apply bug fixes. If those fixes are already applied in the master branch,
    # you can do `git cherry-pick <commit#>`. If the commit was a merge use "-m 1".
    promptForUser = ("Step 3: Are the bug fixes you want to apply currently in the master branch (y or n)? ")
    bugFixesInMaster = verify_user_input(promptForUser)
    if bugFixesInMaster == "y":
        bugFixesCommits = input(f"{spacesForFormatting} List the commits (comma separted) that contain bug fixes: ")
        promptForUser = (f"{spacesForFormatting} Are these commits merges (y or n)? ")
        commitsAreMerged = verify_user_input(promptForUser)
        for commitId in bugFixesCommits.split(","):
            if commitsAreMerged == "y":
                if os.system(f"git cherry-pick -m 1 {commitId}") != 0:
                    input(f"""`git cherry-pick -m 1 {commitId}` was not applied successfully. Suspend this script (ctrl + z) and follow the steps listed above this message in yellow.
Once completed, on the command line run `jobs`, locate papi_procedures.py, and finally run `fg %# where # is the value in brakcets in the left most column.`""")
            else:
                if os.system(f"git cherry-pick {commitId}")  != 0:
                    input(f"""`git cherry-pick {commitId}` was not applied successfully. Suspend this script (ctrl + z) and follow the steps listed above this message in yellow.
Once completed, on the command line run `jobs`, locate papi_procedures.py, and finally run `fg %# where # is the value in brakcets in the left most column.`""")
    elif bugFixesInMaster == "n":
        input(f"{spacesForFormatting} Suspend this script (ctrl + z) and apply the bug fixes that are necessary. Once the bug fixes have been COMMITTED, on the command line run `jobs`\n"
              f"{spacesForFormatting} locate papi_procedures.py, and finally run `fg %#` where # is the value in brackets in the left most column.")

    print(" ")
    # Step 4, build and test your changes on different platforms.
    input("Step 4: Your bug fixes have been applied. You need to now test these changes on different platforms. Press ctrl + z to suspend this script.\n"
          f"{spacesForFormatting} Once testing has been completed, on the command line run `jobs`, locate papi_rocedures.py, and finally run `fg %#` where # is the value in brackets in the left most column.")

    print(" ")
    # Step 5, create the tag papi-X-Y-Z-N-t (N is an incremental bug fix identifier).
    bugFixTag = input("Step 5: Provide a tag with the formatting being papi-X-Y-Z-N-t (e.g. papi-7-2-0-1-t, note that N is the incremental bug fix identifier): ")
    os.system(f"git tag -a {bugFixTag}")

    print(" ")
    # Step 6, pushing changes.
    input("Step 6: Changes will now be pushed. (Press Enter to Continue).")
    os.system("git push --tags")

    print(" ")
    # Step 7, verify the tag was created.
    input(f"Step 7: Go to https://github.com/icl-utk-edu/papi/tags and verify your tag {bugFixTag} was successfully created. (Press Enter to Continue).")

    print(" ")
    # Step 8, create a fresh clone of the papi repository under a directory name including the release number.
    # This release number needs to be the same as provided in Step 5 minus the -t and using periods instead of dashes. Example: papi-7.2.0.1.
    tarballDirBugFix = input("Step 8: Provide a name to clone PAPI under (e.g. papi-7.2.0.1, note that it must have the release number with the bug identifier): ")
    os.system(f"git clone https://github.com/icl-utk-edu/papi.git {tarballDirBugFix}")

    print(" ")
    # Step 9, tar, zip, and update the permissions.
    input(f"Step 9: The directory {tarballDirBugFix} will now be tar'd and zipped. (Press Enter to Continue).")
    os.system(f"""tar -cvf {tarballDirBugFix}.tar {tarballDirBugFix};
                  gzip {tarballDirBugFix}.tar;
                  chmod 664 {tarballDirBugFix}.tar.gz""")

    print(" ")
    # Step 10, copy the tarball to the website.
    input("Step 10: The tarball will be copied to the website. (Press Enter to Continue).")
    username = getpass.getuser()
    pathToDownloads = f"/home/{username}/websites/icl.utk.edu/projects/papi/downloads/"
    os.system(f"scp {tarballDirBugFix}.tar.gz {username}@icl.utk.edu:{pathToDownloads}")

    print(" ")
    # Step 11, check the newly created link is functioning properly and can be downloaded.
    input("Step 11: The newly created link with the tarball will now be downloaded to verify it is functioning properly. (Press Enter to Continue).")
    os.system(f"wget http://icl.utk.edu/projects/papi/downloads/{tarballDirBugFix}.tar.gz")
    promptForUser = (f"{spacesForFormatting} Was the tarball successfully downloaded (y or n)? ")
    tarballDowloadSuccess = verify_user_input(promptForUser)
    if tarballDownloadSuccess == "n":
        input(f"{spacesForFormatting} Press ctrl + z to suspend this script. Once the tarball is successfully downloaded, on the command line run\n"
              f"{spacesForFormatting} `jobs`, locate papi_rocedures.py, and finally run `fg %#` where # is the value in brackets in the left most column.")

    print(" ")
    # Step 12, create a link with supporting text on the PAPI software web page.
    # Create an entry on the PAPI wiki: https://github.com/icl-utk-edu/papi/wiki/PAPI-Releases
    # announcing the new release with details about the changes.
    input("Step 12: Create a link with supporting text on the PAPI software web page and create an entry on the PAPI wiki (https://github.com/icl-utk-edu/papi/wiki/PAPI-Releases). (Press Enter to Continue).")

    print("\033[32mThe bug fix procedures have been completed!\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "papi_procedures.py",
                                     description = "Python script to automate the PAPI procedures e.g. release or bug fix."
                                    )
    parser.add_argument("--procedure",
                        required = True,
                        choices = ["release", "bugfix"],
                        help = "Select a procedure to begin, options include: release or bugfix"
                       )

    args = parser.parse_args()
    if args.procedure == "release":
        papi_release()
    else:
        papi_bugfix()

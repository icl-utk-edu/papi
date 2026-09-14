**[PAPI: The Performance Application Programming Interface](https://icl.utk.edu/exa-papi/)**

**[Innovative Computing Laboratory (ICL)](http://www.icl.utk.edu/)**

**[PAPI Wiki - Documentation](https://github.com/icl-utk-edu/papi/wiki/)**

**University of Tennessee, Knoxville (UTK)**


***
[TOC]
***

# About

The Performance Application Programming Interface (PAPI) provides tool
designers and application engineers with a consistent interface and methodology
for the use of low-level performance counter hardware found across the entire
compute system (i.e. CPUs, GPUs, on/off-chip memory, interconnects, I/O system,
energy/power, etc.). PAPI enables users to see, in near real time, the
relations between software performance and hardware events across the entire
computer system.

[The ECP Exa-PAPI project](https://icl.utk.edu/exa-papi/) builds on the latest
PAPI project and extends it with:

* Performance counter monitoring capabilities for new and advanced ECP
  hardware, and software technologies.
* Fine-grained power management support.
* Functionality for performance counter analysis at "task granularity" for
  task-based runtime systems.
* "Software-defined Events" that originate from the ECP software stack and are
  currently treated as black boxes (i.e., communication libraries, math
  libraries, task-based runtime systems, etc.)

The objective is to enable monitoring of both types of performance
events---hardware- and software-related events---in a uniform way, through one
consistent PAPI interface. Third-party tools and application developers will
have to handle only a single hook to PAPI in order to access all hardware
performance counters in a system, including the new software-defined events.


***

# Preferred Citation

If PAPI is used in a project or publication, please cite the following paper:

> Jagode H, Danalis A, Congiu G, Barry D, Castaldo A, Dongarra J. 
> **Advancements of PAPI for the exascale generation.**
> *The International Journal of High Performance Computing Applications.* 
> 2024;39(2):251-268. 
> [doi:10.1177/10943420241303884](https://journals.sagepub.com/doi/10.1177/10943420241303884)

**This helps us track the impact of the project and supports ongoing development. Thank you!**


***

# Getting Help

* Visit our FAQ at: <https://icl-utk-edu.github.io/papi/PAPI_FAQ.html> 
  or read a snapshot of the FAQ in papi/PAPI_FAQ.html
* For assistance with PAPI, email ptools-perfapi@icl.utk.edu.
* You can also join the PAPI User Google group by going to
  <https://groups.google.com/a/icl.utk.edu/forum/#!forum/ptools-perfapi> 
  to read historical postings to the list.

***


# PAPI's Merge Model

PAPI adopts the git merge model. This means that new features get integrated into the history through merge commits. A clean history would look as follows:
```
*   185dcb38b Merge pull request #xxx from neo/rocm_feature_branch
|\
| * efdcf6512 rocm: add tests for introduced changes
| * 55ebf1416 rocm: make changes to rocm component
|/
*   b2b142317 Merge pull request #yyy from morpheus/cuda_feature_branch
|\
| * 2e7a0bbe4 cuda: add matrix multiplication test
|/
*   220b0f28e Merge pull request #zzz from trinity/libpfm4_udpate_branch
|\
| * 42df8d271 libpfm4: update to latest master
|/
*   2cac08381 Merge pull request #xyz ...
```

The above history is the result of rebasing feature branches onto the PAPI master branch before merging. To do this, please follow the steps below:

1. From your local repository, add the PAPI repository as a remote (i.e. `git remote add upstream https://www.github.com/icl-utk-edu/papi.git`).

2. Switch to your local master branch and then run `git fetch upstream master && git reset --hard FETCH_HEAD`.

3. Switch back to your feature branch and then run `git rebase master`.

4. If conflicts are detected during the rebase then resolve them. Once resolved, add the changes to the staging area (i.e. `git add -u`) and continue rebasing (i.e. `git rebase --continue`).

5. Lastly, push your rebased feature branch to the remote (i.e. `git push -f origin feature_branch`).

***

# Contributing as a Developer

The PAPI project welcomes contributions from new developers. Contributions can
be offered through the standard GitHub pull request model. We strongly
encourage you to coordinate large contributions with the PAPI development team
early in the process.

**For timely pull request reviews and feedback, it is important to submit 
one (1) pull request per feature / bug fix.**

In order to create a pull request on a public read-only repo, 
you will need to do the following:

1. Fork the PAPI repository:
    - Go to https://github.com/icl-utk-edu/papi.
    - Click **Fork** in the upper right corner.
    - Provide an optional repository name.
    - Click **Create fork** in the bottom right corner.

2. Clone it (i.e. `git clone https://github.com/$USERNAME/$FORKNAME.git` where `$USERNAME` is your GitHub username and `$FORKNAME` is the name of your created fork).

3. Create a feature branch for every feature regardless of how small (from branch `master` run `git checkout -b feature_branch`).

4. Make your changes locally and push them to remote (i.e. `git push -u origin feature_branch`).

5. Go to https://github.com/icl-utk-edu/papi and create a pull request. As a push was just made a yellow banner will appear right above the green **Code** dropdown button. If this is not the case then follow the steps outlined below:
    - Click on the **Pull requests** button located in the top left corner.
    - Click on the green **New pull request** button located in the middle right corner.
    - Click on the blue **compare across forks** button located below the **Compare changes** heading.
    - For the head repository select your PAPI fork and then for compare select the branch that you have made changes on.
    - Lastly, review the changes and if everything looks correct click **Create pull request**.

***

# Contributing as a Reviewer

A PAPI team member may ask you to review/test a pull request if one of the below conditions are met:
1. The pull request addresses an issue you opened in the PAPI [repository](https://github.com/icl-utk-edu/papi/issues).
2. The pull request addresses an email you sent to either the PAPI [developers](https://groups.google.com/a/icl.utk.edu/g/perfapi-devel) or [users](https://groups.google.com/a/icl.utk.edu/g/ptools-perfapi) mailing lists.

Testing the PR can be done via [GitHub CLI](#using-github-cli) or [Git CLI](#using-git-cli), see below for steps.

## Using GitHub CLI

Two options exist to test a PR via GitHub CLI:

Option 1: You are already in a GitHub repository
```
# In a PAPI repository (i.e. git clone https://github.com/icl-utk-edu/papi.git).
gh pr checkout $NUMBER (where $NUMBER is the PR number)
# Not in a PAPI repository.
gh pr checkout $NUMBER (where $NUMBER is the PR number) --repo https://github.com/icl-utk-edu/papi.git
```

Option 2: You are not already in a GitHub repository
```
git clone https://github.com/icl-utk-edu/papi.git
cd papi
gh pr checkout $NUMBER (where $NUMBER is the PR number)
```

If your system does not have GitHub CLI available and this is your preferred method of reviewing a PR
then see the [installation options](https://github.com/cli/cli#installation) provided by GitHub.

## Using Git CLI

Two options exist to test a PR via Git CLI:

Option 1: You are already in a GitHub repository
```
# In a PAPI repository (i.e. git clone https://github.com/icl-utk-edu/papi.git).
git fetch origin/$NUMBER/head (where $NUMBER is the PR number)
git checkout FETCH_HEAD
# Not in a PAPI repository.
git remote add papi https://github.com/icl-utk-edu/papi.git
git fetch papi/$NUMBER/head (where $NUMBER is the PR number)
git checkout FETCH_HEAD
```

Option 2: You are not already in a GitHub repository
```
git clone https://github.com/icl-utk-edu/papi.git
cd papi
git fetch origin pull/$NUMBER/head (where $NUMBER is the PR number)
git checkout FETCH_HEAD
```
***

# Resources

* Visit the [Exa-PAPI website](https://icl.utk.edu/exa-papi/) to find out more
  about ongoing PAPI and
  [PAPI++](https://www.exascaleproject.org/papi-as-de-facto-standard-interface-for-performance-event-monitoring-at-the-exascale/)
  developments and research.
* Visit the [PAPI website (retired)](https://icl.utk.edu/papi/) for basic
  information about PAPI.
* Visit the [ECP website](https://www.exascaleproject.org/) to find out more
  about the DOE Exascale Computing Initiative.
* Visit the [PAPI Papers and Presentations](https://www.icl.utk.edu/view/biblio/project/papi?items_per_page=All) to find out more about PAPI papers and presentations.

***


# License

    Copyright (c) 2026, Innovative Computing Laboratory,
    University of Tennessee Knoxville
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the University of Tennessee nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    This open source software license conforms to the BSD License template.

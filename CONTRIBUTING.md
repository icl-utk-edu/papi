# Contributing to the PAPI Library
Thank you for contributing to the PAPI library! This guide will outline reporting security vulnerabilities, development workflows, and contribution standards.

## :clipboard: Table of Contents
* [:lock: Reporting Security Vulnerabilities](#lock-reporting-security-vulnerabilities)
* [:open_file_folder: PAPI Directory Structure](#open_file_folder-papi-directory-structure)
* [:handshake: Contributing to the PAPI Codebase](#handshake-contributing-to-the-papi-codebase)
    * [:computer: Contributing as a Developer](#computer-contributing-as-a-developer)
    * [:eyeglasses: Contributing as a Reviewer](#eyeglasses-contributing-as-a-reviewer)
* [:spiral_notepad: Pull Request Guidelines for Developers](#spiral_notepad-pull-request-guidelines-for-developers)
* [:left_speech_bubble: Developer Communications](#left_speech_bubble-developer-communications)

## :lock: Reporting Security Vulnerabilities
PAPI maintainers take security seriously. If you happen to discover a security vulnerability then please immediately report the vulnerability through GitHub's [Report a vulnerabiity](https://github.com/icl-utk-edu/papi/security/advisories/new) feature.

For additional information on PAPI's security policy, please see the [SECURITY.md](https://github.com/icl-utk-edu/papi/blob/master/SECURITY.md).

## :open_file_folder: PAPI Directory Structure
> [!NOTE]
>
> The below PAPI directory structure is non-exhaustive and serves as a point of reference for commonly
> updated directories and source files.
```text
papi/
├── .github/                         # Includes CI workflows and scripts.
└── src/                             # PAPI framework (including papi.c and papi.h).
    ├── components/                  # Source code for PAPI components.
    │   ├── amd_smi/
    │   ├── cuda/
    │   └── rocp_sdk/                # ...and many more.
    ├── counter_analysis/            # Source code for 
    └── utils/                       # PAPI utilities.
    │    ├── papi_avail.c
    │    ├── papi_command_line.c
    │    ├── papi_component_avail.c
    │    └── papi_native_avail.c     # ...and many more.
    ├── ctests/                      # Tests written in C to verify PAPI's C-API.
    └── ftests/                      # Tests written in Fortran to verify PAPI's Fortan API.
```

## :handshake: Contributing to the PAPI Codebase

### :computer: Contributing as a Developer

PAPI is an open source project and welcomes contributions from all developers. To allow this, PAPI follows the [forking workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow):
1. Fork the PAPI repository:
    - Go to https://github.com/icl-utk-edu/papi.
    - Click **Fork** in the upper right corner.
    - Provide an optional repository name.
    - Click **Create fork** in the bottom right corner.
2. Clone the created fork:
    ```sh
    # Where $USERNAME is your GitHub username and $FORKNAME is the name of your created fork.
    git clone https://github.com/$USERNAME/$FORKNAME.git
    ```

3. Create a feature branch for every feature regardless of how small:
    ```sh
    # Where $FEATUREBRANCH is the name of your feature branch.
    git checkout -b $FEATUREBRANCH
    ```

4. Make your changes locally and push them to remote:
    ```sh
    # Where $FEATUREBRANCH is the name of the feature branch created in the step above.
    git push -u origin $FEATUREBRANCH
    ```
5. Create a pull request.

### :eyeglasses: Contributing as a Reviewer

A PAPI team member may ask you to review/test a pull request if one of the below conditions are met:
1. The pull request addresses an issue you opened in the PAPI [repository](https://github.com/icl-utk-edu/papi/issues).
2. The pull request addresses an email you sent to either the PAPI [developers](https://groups.google.com/a/icl.utk.edu/g/perfapi-devel) or [users](https://groups.google.com/a/icl.utk.edu/g/ptools-perfapi) mailing lists.

Testing the PR can be done either via Option A or Option B shown below.

#### Option A: Using GitHub CLI
---

On one hand, if you are already in a GitHub repository, then the steps are:
```sh
# In a PAPI repository (i.e. git clone https://github.com/icl-utk-edu/papi.git).
gh pr checkout $NUMBER # Where $NUMBER is the PR number.
# Not in a PAPI repository.
gh pr checkout $NUMBER --repo https://github.com/icl-utk-edu/papi.git # Where $NUMBER is the PR number.
```

On the other hand, if you are not already in a GitHub repository, then the steps become:
```sh
git clone https://github.com/icl-utk-edu/papi.git
cd papi
gh pr checkout $NUMBER # Where $NUMBER is the PR number.
```

If your system does not have GitHub CLI available and this is your preferred method of reviewing a PR then see the [installation options](https://github.com/cli/cli#installation) provided by GitHub.

#### Option B: Using Git CLI
---

On one hand, if you are already in a GitHub repository, then the steps are:
```sh
# In a PAPI repository (i.e. git clone https://github.com/icl-utk-edu/papi.git).
git fetch origin/$NUMBER/head # Where $NUMBER is the PR number.
git checkout FETCH_HEAD
# Not in a PAPI repository.
git remote add papi https://github.com/icl-utk-edu/papi.git
git fetch papi/$NUMBER/head # Where $NUMBER is the PR number.
git checkout FETCH_HEAD
```

On the other hand, if you are not already in a GitHub repository, then the steps become:
```sh
git clone https://github.com/icl-utk-edu/papi.git
cd papi
git fetch origin pull/$NUMBER/head # Where $NUMBER is the PR number.
git checkout FETCH_HEAD
```

## :spiral_notepad: Pull Request Guidelines for Developers

For timely pull request reviews and feedback, it is advised to submit one pull request per feature/bug fix.

### :pencil2: Branch Names
> [!NOTE]
>
> Aim to keep branch names concise, but descriptive to reflect the purpose of the branch.

When creating a branch for your work, use the following naming convention to make branch names both informative and consistent: `month-day-year-{component name, filename, etc.}-short-description-of-change`.

For example, an update to the `cuda` component's enumeration logic was done on October 5, 2026; therefore, a resulting branch name could be:
```
10-05-2026-cuda-fixed-enumeration-logic
```

### :twisted_rightwards_arrows: Keeping Your Branch in Sync

To keep your branch updated with the latest changes in the PAPI master branch:
```sh
# Add a remote for PAPI master.
git remote add upstream https://www.github.com/icl-utk-edu/papi.git
# From the local master branch.
git fetch upstream master && git reset --hard FETCH_HEAD
# Switch back to your feature branch.
git checkout $FEATUREBRANCH
# Rebase the feature branch.
git rebase master
```

> [!NOTE]
>
> If conflicts are detected during rebasing then resolve them, add the changes to
> staging, and continuing rebasing.

### :label: Labeling

Once a PR has been opened, it is advised to add labels for classification.

The table below outlines a subset of available label group's for the PAPI repository:
| Label Group  |  Description |
| ------------- | ------------- |
| ![component](https://img.shields.io/badge/component-A1AEB1)  | PRs related to PAPI components  |
| ![type](https://img.shields.io/badge/type-BF9FF2)  | The type of PR (e.g. bug, refactoring, maintenance)  |
| ![update](https://img.shields.io/badge/update-4682B4)  | Updates to the PAPI framework (e.g. documentation, presets, build system) |

## :left_speech_bubble: Developer Communications
Communication with a PAPI maintainer can be done by:
1. [GitHub Issues](https://github.com/icl-utk-edu/papi/issues)
2. [Mailing list](https://groups.google.com/a/icl.utk.edu/g/ptools-perfapi) for PAPI users.
3. [Mailing list](https://groups.google.com/a/icl.utk.edu/g/perfapi-devel) for PAPI developers.
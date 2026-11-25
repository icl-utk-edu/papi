# Contributing to the PAPI Library
Thank you for contributing to the PAPI library! This guide outlines how to report security vulnerabilities, follow our development workflows, and meet our contribution standards.

## :clipboard: Table of Contents
* [:lock: Reporting Security Vulnerabilities](#lock-reporting-security-vulnerabilities)
* [:open_file_folder: PAPI Directory Structure](#open_file_folder-papi-directory-structure)
* [:handshake: Contributing to the PAPI Codebase](#handshake-contributing-to-the-papi-codebase)
    * [:computer: Contributing as a Developer](#computer-contributing-as-a-developer)
    * [:eyeglasses: Contributing as a Reviewer](#eyeglasses-contributing-as-a-reviewer)
* [:spiral_notepad: Pull Request Guidelines for Developers](#spiral_notepad-pull-request-guidelines-for-developers)
* [:left_speech_bubble: Developer Communications](#left_speech_bubble-developer-communications)

## :lock: Reporting Security Vulnerabilities
PAPI maintainers take security seriously. If you happen to discover a security vulnerability then please immediately report the vulnerability through GitHub's [Report a vulnerability](https://github.com/icl-utk-edu/papi/security/advisories/new) feature.

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
    ├── counter_analysis/            # Performance counter analysis tools. 
    ├── utils/                       # PAPI utilities.
    │    ├── papi_avail.c
    │    ├── papi_command_line.c
    │    ├── papi_component_avail.c
    │    └── papi_native_avail.c     # ...and many more.
    ├── ctests/                      # Tests for PAPI's C API.
    └── ftests/                      # Tests for PAPI's Fortran API.
```

## :handshake: Contributing to the PAPI Codebase

### :computer: Contributing as a Developer

PAPI is an open source project and welcomes contributions from all developers. PAPI follows the [forking workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow):
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

3. Change into the cloned repository:
   ```sh
    cd $FORKNAME
    ```

4. Create a branch for each feature or bug fix, regardless of size:
    ```sh
    # Where $BRANCH is the name of your branch.
    git checkout -b $BRANCH
    ```

5. Make your changes locally.

6. Stage and commit your changes:
    ```sh
    git add <files>
    git commit -m "Desribe your changes"
    ```

7. Push committed changes to your fork:
    ```sh
    # Where $BRANCH is the name of the branch created in the step above.
    git push -u origin $BRANCH
    ```

8. Open a pull request from the new branch to the PAPI repository.

### :eyeglasses: Contributing as a Reviewer

A PAPI team member may ask you to review/test a pull request if one of the following conditions are met:
1. The pull request addresses an issue you opened in the PAPI [repository](https://github.com/icl-utk-edu/papi/issues).
2. The pull request addresses an email you sent to either the PAPI [developers](https://groups.google.com/a/icl.utk.edu/g/perfapi-devel) or [users](https://groups.google.com/a/icl.utk.edu/g/ptools-perfapi) mailing lists.

#### Reviewer Responsibilities
Reviewers should evaluate the correctness of the proposed changes, test the affected functionality where possible,
identify potential regressions, and provide constructive feedback. When reporting test results,
reviewers should include relevant hardware and software configuration details.

#### Testing the Pull Request
Testing the PR can be done either via Option A or Option B shown below.

##### Option A: Using GitHub CLI
---

If you are already working in a Git repository, then the steps are:
```sh
# In a PAPI repository (i.e. git clone https://github.com/icl-utk-edu/papi.git).
gh pr checkout $NUMBER # Where $NUMBER is the PR number.
# Not in a PAPI repository.
gh pr checkout $NUMBER --repo https://github.com/icl-utk-edu/papi.git # Where $NUMBER is the PR number.
```

If you do not have a local Git repository, then the steps become:
```sh
git clone https://github.com/icl-utk-edu/papi.git
cd papi
gh pr checkout $NUMBER # Where $NUMBER is the PR number.
```

If your system does not have GitHub CLI available and this is your preferred method of reviewing a PR then see the [installation options](https://github.com/cli/cli#installation) provided by GitHub.

##### Option B: Using Git CLI
---

If you are already working in a local Git repository, then the steps are:
```sh
# In a PAPI repository (i.e. git clone https://github.com/icl-utk-edu/papi.git).
git fetch origin pull/$NUMBER/head # Where $NUMBER is the PR number.
git checkout FETCH_HEAD
# Not in a PAPI repository.
git remote add papi https://github.com/icl-utk-edu/papi.git
git fetch papi pull/$NUMBER/head # Where $NUMBER is the PR number.
git checkout FETCH_HEAD
```

If you do not have a local Git repository, then the steps become:
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
# Add a remote for PAPI master (only needed once).
git remote add upstream https://www.github.com/icl-utk-edu/papi.git
# Fetch the latest changes from PAPI.
git fetch upstream
# Switch to the branch you wish to update.
git checkout $BRANCH
# Rebase onto PAPI's master branch.
git rebase upstream/master
```

> [!NOTE]
>
> If conflicts are detected during rebasing then resolve them, add the changes to
> staging, and continue the rebase.

### :label: Labeling
Once a PR has been opened, PAPI maintainers will assign appropriate labels to help
classify and track the contribution.

## :left_speech_bubble: Developer Communications
You can contact PAPI maintainers through:
1. [GitHub Issues](https://github.com/icl-utk-edu/papi/issues)
2. [Mailing list](https://groups.google.com/a/icl.utk.edu/g/ptools-perfapi) for PAPI users.
3. [Mailing list](https://groups.google.com/a/icl.utk.edu/g/perfapi-devel) for PAPI developers.
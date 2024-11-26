As of now, the GitHub CI is designed to run in three instances:
1. A per component basis, meaning if a component's codebase is updated then we only will run CI tests for that component. As an example, if we update `cupti_profiler.c` in `src/components/cuda` then we will only run CI tests for that component. Note that this includes updates to subdirectories located in a component's directory (e.g. `src/components/cuda/tests`).
2. A change to the Counter Analysis Toolkit i.e. in the `src/counter_analysis_toolkit` directory and any subdirectories.
3. A change in the PAPI framework i.e. in the `src/` directory (excluding individual components and the Counter Analysis Toolkit). If this occurs then we will run a full test suite.


# Per Component Basis
 All per component basis tests have a `.yml` that is structured with `componentName_component.yml`. As 
an example for the `cuda` component we would have a `.yml` of `cuda_component.yml`. Therefore,
if a new component is added to PAPI, you will need to create a `.yml` based on the aforementioned structure.

Along with creating the `.yml` file, you will need to add an associated workflow. Below is a skeleton that can
be used as a starting point. As a reminder, make sure to change the necessary fields out for your component.

```yml
name: cuda # replace cuda with your component name

on:
  pull_request:
    paths:
      - 'src/components/cuda/**' # replace the cuda path with your component

jobs:
  component_tests:
    strategy:
      matrix:
        component: [cuda] # replace cuda with your component name
        debug: [yes, no] 
        shlib: [with, without]
      fail-fast: false
    runs-on: [self-hosted, nvidia_gpu]
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - name: cuda component tests # replace cuda with your component name
        run: .github/workflows/ci_per_component.sh ${{matrix.component}} ${{matrix.debug}} ${{matrix.shlib}}
```` 

# Counter Analysis Toolkit
The Counter Analysis Toolkit (CAT) CI uses the `cat.yml` and `ci_cat.sh` files. Any updates to the CI for CAT need to be done in these two files.

# PAPI Framework
The PAPI framework CI uses the `papi_framework.yml` and `ci_papi_framework.sh` files. Any updates to the CI for the PAPI framework need to be done in these two files.

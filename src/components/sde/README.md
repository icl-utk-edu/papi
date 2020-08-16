# SDE Component

The SDE component enables PAPI to read Software Defined Events (SDEs).

* [Enabling the SDE Component](#markdown-header-enabling-the-sde-component)
* [Environment Variables](#markdown-header-environment-variables)
* [FAQ](#markdown-header-faq)

***
## Enabling the SDE Component

To enable reading of SDE counters the user needs to link against a
PAPI library that was configured with the SDE component enabled. As an
example the following command: `./configure --with-components="sde"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## Environment Variables

For SDE, PAPI requires one environment variable: **SDE\_LIB\_PATHS**. 

This is required at runtime and for `papi_native_avail` that needs to know where to look for libraries with SDE support.

Example:

	export SDE_LIB_PATHS=${SDE_TEST_PATH}/tests/libGamum.so

Set the variable SDE\_LIB\_PATHS to contain the paths to all the libraries in your system with SDE support.

## FAQ

* [Software Defined Events](https://bitbucket.org/icl/papi/wiki/Software_Defined_Events.md)

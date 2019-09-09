# LMSENSORS Component
The PAPI lmsensors component requires lmsensors version >= 3.0.0.


## Installing PAPI with LMSENSORS Component
There is ONE required environment variable: `PAPI_LMSENSORS_ROOT`. This is
required for both compiling, and at runtime. 

An example that works on ICL's Saturn system (at this writing):

    export PAPI_LMSENSORS_ROOT=/usr

Within `PAPI_LMSENSORS_ROOT`, we expect the following standard directories:

* `PAPI_LMSENSORS_ROOT/include` or `PAPI_LMSENSORS_ROOT/include/sensors`
* `PAPI_LMSENSORS_ROOT/lib64`


For a standard installed system, this is the only environment variable
required for both compile and runtime. 

System configurations can vary. Some systems use Spack, a package
manager, to automatically keep paths straight. Others require
"module load" commands to provide some services, e.g.
"module load lmsensors", and these may also set environment
variables and change the `LD_LIBRARY_PATH` search order.

Users may require the help of sysadmin personnel to navigate these
facilities and gain access to the correct libraries.

### Configure PAPI with LMSENSORS Enabled

We presume you have navigated to the
directory papi/src, AND that you have exported `PAPI_LMSENSORS_ROOT`. 

In the papi/src directory:

    ./configure --with-components="lmsensors"
    make


### Testing PAPI with LMSENSORS Enabled

From papi/src:

    utils/papi_component_avail


For the LMSENSORS component to be operational, it must find the dynamic
library `libsensors.so`.

If it is not found (or is not functional) then the component will be
listed as "disabled" with a reason explaining the problem. If library
was not found, then it is not in the expected place.  The component
can be configured to look for the library in a specific place, and
using an alternate name if desired. Detailed instructions are
contained in the `Rules.lmsensors` file.  They are technical, users may wish
to enlist the help of a sysadmin.

### List LMSENSORS Supported Events
From papi/src:

    utils/papi_native_avail | grep -i sensors
    
## Author
* Frank Winkler (frank.winkler@icl.utk.edu)
* Anthony Castaldo (tonycastaldo@icl.utk.edu)
* Dan Terpstra (terpstra@icl.utk.edu)

# INFINIBAND_UMAD Component
The PAPI infiniband component enables PAPI-C to access hardware monitoring counters for InfiniBand devices through the OFED library. Since a new interface was introduced with OFED version 1.4 (released Dec 2008), the current InfiniBand component does not support OFED versions < 1.4.


## Installing PAPI with INFINIBAND_UMAD Component
There is ONE required environment variable: `PAPI_INFINIBAND_UMAD_ROOT`. This is
required for both compiling, and at runtime. 

An example that works on ICL's Saturn system (at this writing):

    export PAPI_INFINIBAND_UMAD_ROOT=/usr

Within `PAPI_INFINIBAND_UMAD_ROOT`, we expect the following standard directories:

* `PAPI_INFINIBAND_UMAD_ROOT/include`
* `PAPI_INFINIBAND_UMAD_ROOT/lib64`


For a standard installed system, this is the only environment variable
required for both compile and runtime. 

System configurations can vary. Some systems use Spack, a package
manager, to automatically keep paths straight. Others require
"module load" commands to provide some services, e.g.
"module load infiniband", and these may also set environment
variables and change the `LD_LIBRARY_PATH` search order.

Users may require the help of sysadmin personnel to navigate these
facilities and gain access to the correct libraries.

### Configure PAPI with INFINIBAND_UMAD Enabled

We presume you have navigated to the
directory papi/src, AND that you have exported `PAPI_INFINIBAND_UMAD_ROOT`. 

In the papi/src directory:

    ./configure --with-components="infiniband_umad"
    make

### Testing PAPI with INFINIBAND_UMAD Enabled

From papi/src:

    utils/papi_component_avail

For the INFINIBAND_UMAD component to be operational, it must find the dynamic
libraries `libibmad.so` and `libibumad.so`.

If it is not found (or is not functional) then the component will be
listed as "disabled" with a reason explaining the problem. If library
was not found, then it is not in the expected place.  The component
can be configured to look for the library in a specific place, and
using an alternate name if desired. Detailed instructions are
contained in the `Rules.infiniband_umad` file.  They are technical, users may wish
to enlist the help of a sysadmin.

### List INFINIBAND_UMAD Supported Events
From papi/src:

    utils/papi_native_avail | grep -i infiniband

## Author
* Frank Winkler (frank.winkler@icl.utk.edu)
* Anthony Castaldo (tonycastaldo@icl.utk.edu)
* Dan Terpstra (terpstra@icl.utk.edu)

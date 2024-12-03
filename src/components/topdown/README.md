# TOPDOWN Component

The `topdown` component enables accessing the `PERF_METRICS` Model Specific
Register (MSR) of modern Intel PMUs, and makes it simple to properly 
interpret the results.

* [Enabling the TOPDOWN Component](#enabling-the-topdown-component)
* [Adding More Architectures](#adding_more_architectures)

## Enabling the TOPDOWN Component

To enable reading of topdown metrics the user needs to link against a
PAPI library that was configured with the topdown component enabled. As an
example the following command: `./configure --with-components="topdown"` is
sufficient to enable the component.

## Interpreting Results

The events added by this component ending in "_PERC" should be cast to double 
values in order to be properly interpreted as percentages. An example of how
to do so follows:

	PAPI_start(EventSet);
	
	/* some block of code... */
	
	PAPI_stop(EventSet, values);
	
	printf("First metric was %.1f\n", *((double *)(&values[0])));

## Adding More Architectures

To contribute more supported architectures to the component, add the cpuid model
of the architecture to the case statement in `_topdown_init_component` of 
[topdown.c](./topdown.c) and set the relevant options (`supports_l2`, etc.)
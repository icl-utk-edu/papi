# PERF\_EVENT\_UNCORE Component

The PERF\_EVENT_UNCORE component enables PAPI to access perf\_event CPU uncore and northbridge.

* [Enabling the PERF\_EVENT\_UNCORE Component](#markdown-header-enabling-the-perf-event-uncore-component)
* [FAQ](#markdown-header-faq)

***
## Enabling the PERF\_EVENT\_UNCORE Component

This component is enabled by default.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## FAQ

1. [Measuring Uncore Events](#markdown-header-measuring-uncore-events)

## Measuring Uncore Events

The read counters of uncore events requires to specify the CPU identifier such as **`:cpu=0`**.

For example, to read counters from the native uncore event `hswep_unc_ha0::UNC_H_RING_AD_USED:CW` on Haswell:

	papi_command_line hswep_unc_ha0::UNC_H_RING_AD_USED:CW:cpu=0
	
**Hint**: Use `lscpu` on the respective node to get the distribution of CPU identifiers across the sockets.

Example for a dual-socket Intel Haswell node with 24 physical cores:

	lscpu
	Architecture:          x86_64
	CPU op-mode(s):        32-bit, 64-bit
	Byte Order:            Little Endian
	CPU(s):                24
	...
	NUMA node0 CPU(s):     0-11
    NUMA node1 CPU(s):     12-23

The last two lines show you the distribution of CPU identifiers for socket1 (NUMA node0) and socket2 (NUMA node1).
If you want to measure uncore events from socket1, you can use the CPU identifiers from 0-11.




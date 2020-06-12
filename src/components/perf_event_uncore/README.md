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

The read counters of uncore events requires to specify the CPU/socket identifier such as **`:cpu=0`**.

For example, to read counters from the native uncore event `hswep_unc_ha0::UNC_H_RING_AD_USED:CW` on Haswell:

	papi_command_line hswep_unc_ha0::UNC_H_RING_AD_USED:CW:cpu=0
	
**Hint**: Use `lscpu` on the respective compute node to get the socket information per CPU.

Example for a dual-socket Intel Haswell node with 24 physical cores:

	lscpu
	Architecture:          x86_64
	CPU op-mode(s):        32-bit, 64-bit
	Byte Order:            Little Endian
	CPU(s):                24
	On-line CPU(s) list:   0-23
	Thread(s) per core:    1
	Core(s) per socket:    12
	...

You can determine the socket identifiers based on the number of physical cores and cores per socket.
Thus,`:cpu=0` and `:cpu=12` are the socket identifiers on Haswell nodes.




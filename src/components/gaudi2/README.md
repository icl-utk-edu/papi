# Gaudi2 Component

The `gaudi2` component provides access to hardware performance counters on Intel Gaudi2 AI Accelerators through the SPMU interface.

- [Environment Variables](#environment-variables)
- [Enabling the Gaudi2 Component](#enabling-the-gaudi2-component)

## Environment Variables
The `gaudi2` component requires setting the `PAPI_GAUDI2_ROOT` environment variable to habanalabs installed directory for `hl-thunk` headers and libraries.

```bash
export PAPI_GAUDI2_ROOT=/usr`
```

## Enabling the Gaudi2 Component

To enable the `gaudi2` component, configure and build PAPI with the component enabled as follows:

```bash
./configure --with-components="gaudi2"
make && make install
```
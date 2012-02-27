#ifndef PAPI_DATA_H
#define PAPI_DATA_H

extern hwi_presets_t _papi_hwi_presets;

int _papi_hwi_derived_type( char *derived, int *code );
int _papi_hwi_derived_string( int type, char *derived, int len );

#endif

#ifndef PAPI_DATA_H
#define PAPI_DATA_H

extern hwi_presets_t _papi_hwi_presets;
extern const hwi_preset_info_t _papi_hwi_preset_info[PAPI_MAX_PRESET_EVENTS];
extern const unsigned int _papi_hwi_preset_type[PAPI_MAX_PRESET_EVENTS];

int _papi_hwi_derived_type( char *derived, int *code );
int _papi_hwi_derived_string( int type, char *derived, int len );

#endif

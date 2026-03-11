#!/bin/bash
#Note: It is possible that the listed desired events for each test 
# may not exist on the architecture you are running on or with the 
# ROCm version you are using. If you run with the option --with-desired-events
# and the test fails, verify with utils/papi_native_avail the desired event exists with
# the appended qualifiers.

advanced_desired_events="rocp_sdk:::SQ_CYCLES:device=0,\
rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=0:device=0,\
rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=1:device=0,\
rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=2:device=0,\
rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=3:device=0,\
rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0,\
rocp_sdk:::SQ_WAVE_CYCLES:DIMENSION_SHADER_ENGINE=0:device=0,\
rocp_sdk:::SQ_WAVE_CYCLES:DIMENSION_SHADER_ENGINE=1:device=0,\
rocp_sdk:::SQ_WAVE_CYCLES:DIMENSION_SHADER_ENGINE=2:device=0,\
rocp_sdk:::SQ_WAVE_CYCLES:DIMENSION_SHADER_ENGINE=3:device=0,\
rocp_sdk:::SQ_WAVE_CYCLES:device=0"

echo "make advanced:"
make advanced
if [ "$1" = "--with-desired-events" ]; then
    echo "Running: ./advanced --rocp-sdk-native-event-names ${advanced_desired_events[@]}"
    ./advanced --rocp-sdk-native-event-names "${advanced_desired_events[@]}"
else
    echo "Running: ./advanced"
    ./advanced
fi
echo -e "-------------------------------------"

simple_desired_events="rocp_sdk:::SQ_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=3,\
rocp_sdk:::TCC_CYCLE:device=0:DIMENSION_INSTANCE=2,\
rocp_sdk:::SQ_INSTS:device=0:DIMENSION_INSTANCE=0,\
rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=0,\
rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=1,\
rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=2,\
rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0"

echo "make simple:"
make simple
if [ "$1" = "--with-desired-events" ]; then
    echo "Running: ./simple --rocp-sdk-native-event-names ${simple_desired_events[@]}"
    ./simple --rocp-sdk-native-event-names "${simple_desired_events[@]}"
else
    echo "Running ./simple"
    ./simple
fi
echo -e "-------------------------------------"

simple_sampling_desired_events="rocp_sdk:::SQ_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=3,\
rocp_sdk:::SQ_INSTS:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=0,\
rocp_sdk:::SQ_INSTS:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=1,\
rocp_sdk:::SQ_INSTS:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=2,\
rocp_sdk:::SQ_INSTS:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=3,\
rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=0,\
rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=1,\
rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=2,\
rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0,\
rocp_sdk:::SQ_BUSY_CYCLES:device=1:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=0,\
rocp_sdk:::SQ_BUSY_CYCLES:device=1:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=1,\
rocp_sdk:::SQ_BUSY_CYCLES:device=1:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=2"

echo "make simple_sampling:"
make simple_sampling
if [ "$1" = "--with-desired-events" ]; then
    echo "Running: ./simple_sampling --rocp-sdk-native-event-names ${simple_sampling_desired_events[@]}"
    ./simple_sampling --rocp-sdk-native-event-names "${simple_sampling_desired_events[@]}"
else
    echo "Running ./simple_sampling"
    ./simple_sampling
fi
echo "-------------------------------------"

two_eventsets_desired_events1="rocp_sdk:::SQ_BUSY_CYCLES:device=0,\
rocp_sdk:::SQ_BUSY_CYCLES:device=1,\
rocp_sdk:::TCC_CYCLE:device=1,\
rocp_sdk:::SQ_WAVES:device=0,\
rocp_sdk:::SQ_WAVES:device=1"
two_eventsets_desired_events2="rocp_sdk:::TCC_CYCLE:device=1,\
rocp_sdk:::SQ_INSTS:device=0,\
rocp_sdk:::SQ_INSTS:device=1,\
rocp_sdk:::SQ_BUSY_CYCLES:device=0,\
rocp_sdk:::SQ_BUSY_CYCLES:device=1"

echo "make two_eventsets:"
make two_eventsets
if [ "$1" = "--with-desired-events" ]; then
    echo "Running: ./two_eventsets --first-eventset-native-eventnames ${two_eventsets_desired_events1[@]} --second-eventset-native-eventnames ${two_eventsets_desired_events2[@]}"
    ./two_eventsets --first-eventset-native-eventnames ${two_eventsets_desired_events1[@]} --second-eventset-native-eventnames ${two_eventsets_desired_events2[@]}
else
    echo "Running: ./two_eventsets"
    ./two_eventsets
fi

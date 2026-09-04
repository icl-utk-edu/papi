// C++ STL headers.
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Internal headers.
#include "cuda_range_tests_helper.hpp"
#include "papi.h"
#include "papi_test.h"

/** 
  * @brief Get a cuda_range native event via PAPI_enum_cmp_event.
  *
  *        If the users did not add a cuda_range native event name on the command line then
  *        one will be enumerated for via PAPI_enum_cmp_event. Furthermore, the device qualifier
  *        (i.e device=#) will be used to create the CUDA context on the calling CPU thread.
  *
  * @param cuda_range_cmp_index
  *   The current component index for the cuda_range component.
  * @param &device_index
  *   Stores the device index the cuda_range native event belongs to.
  * @param &cuda_range_native_event_names
  *   A vector to store the enumerated cuda_range native event name. 
*/
void get_cuda_range_native_event_name(int cuda_range_cmp_index, int &device_index, std::vector<std::string> &cuda_range_native_event_names)
{
    // Get the event code for the first cuda_range native event.
    int modifier = PAPI_ENUM_FIRST;
    int cuda_range_eventcode = 0 | PAPI_NATIVE_MASK;
    CHECK_PAPI_API_CALL( PAPI_enum_cmp_event(&cuda_range_eventcode, modifier, cuda_range_cmp_index) );

    // Convert the cuda_range native event code to a cuda_range native event name.
    std::string cuda_range_native_event_name(PAPI_2MAX_STR_LEN, '\0');
    CHECK_PAPI_API_CALL( PAPI_event_code_to_name(cuda_range_eventcode, cuda_range_native_event_name.data()) );
    size_t c_string_length = std::strlen(cuda_range_native_event_name.c_str());
    cuda_range_native_event_name.resize(c_string_length);

    // Post process to get the device index from :device=#.
    std::string qualifier = ":device=";
    size_t device_qualifier_position = cuda_range_native_event_name.find(qualifier);
    if (device_qualifier_position != std::string::npos) {
        std::string device_index_str = cuda_range_native_event_name.substr(device_qualifier_position + qualifier.size(), 1);
        device_index = std::stoi(device_index_str);
    }
    else {
        std::cout << "The cuda_range native event name lacks a device qualifier." << std::endl;
        exit(EXIT_FAILURE);
    }

    cuda_range_native_event_names.push_back(cuda_range_native_event_name);

    return;
}

/** 
  * @brief Get a cuda_range native event via PAPI_enum_cmp_event.
  *
  *        If the users did not add a cuda_range native event name on the command line then
  *        one will be enumerated for via PAPI_enum_cmp_event. PAPI_get_event_info will then
  *        be called to get a cuda_range native event name without a device qualifier. This is done
  *        such that in a component test we can append the device qualifier based on the device count
  *        on the system.
  *
  * @param cuda_range_cmp_index
  *   The current component index for the cuda_range component.
  * @param &cuda_range_native_event_name
  *   Stores the enumerated cuda_range native event name.
*/
void get_cuda_range_native_event_name(int cuda_range_cmp_index, std::string &cuda_range_native_event_name)
{
    // Get the event code for the first cuda_range native event.
    int modifier = PAPI_ENUM_FIRST;
    int cuda_range_eventcode = 0 | PAPI_NATIVE_MASK;
    CHECK_PAPI_API_CALL( PAPI_enum_cmp_event(&cuda_range_eventcode, modifier, cuda_range_cmp_index) );

    PAPI_event_info_t native_event_info;
    CHECK_PAPI_API_CALL( PAPI_get_event_info(cuda_range_eventcode, &native_event_info) );
    cuda_range_native_event_name = native_event_info.symbol;
    
    return;
}

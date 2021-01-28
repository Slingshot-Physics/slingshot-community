#ifndef DATA_MODEL_IO_HEADER
#define DATA_MODEL_IO_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

#include "helper_data_to_json.h"
#include "helper_json_to_data.h"

#include "data_model.h"

data_to_json_f get_jsonifier(const char * type_name);

data_from_json_f get_structifier(const json_string_t * type_name);

// Attempts to write a datamodel type to a file. Returns 1 if successful,
// returns 0 otherwise.
int write_data_to_file(
   const void * data, const char * type_name, const char * filename
);

// Attempts to read a datamodel type from a file. Returns 1 if successful,
// returns 0 otherwise.
int read_data_from_file(void * data, const char * filename);

#ifdef __cplusplus
}
#endif

#endif

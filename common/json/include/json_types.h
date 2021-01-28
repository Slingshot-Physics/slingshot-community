#ifndef JSON_TYPES_HEADER
#define JSON_TYPES_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
   TRUE = 1,
   FALSE = 0,
} K_BOOLEAN;

typedef enum
{
   JSON_OBJECT = 0,
   JSON_ARRAY = 1,
   JSON_STRING = 2,
   JSON_FLOAT_NUMBER = 3,
   JSON_INT_NUMBER = 4,
   JSON_NONE = 100,
} K_VALUETYPE;

typedef enum
{
   PARSE_OBJECT = 0,
   PARSE_OBJECT_KEY = 1,
   PARSE_OBJECT_VALUE = 2,
   PARSE_NUMBER = 3,
   PARSE_FRACTION = 4,
   PARSE_EXPONENT = 5,
   PARSE_STRING = 6,
   PARSE_VALUE = 7,
   PARSE_ESCAPE = 8,
   PARSE_ESCAPE1 = 9,
   PARSE_ESCAPE_UNICODE = 10,
   PARSE_ESCAPE_UNICODE1 = 11,
   PARSE_ESCAPE_UNICODE2 = 12,
   PARSE_ESCAPE_UNICODE3 = 13,
   PARSE_ESCAPE_UNICODE4 = 14,
   PARSE_ARRAY = 15,
   // PARSE_BOOL_TRUE = 15,
   // PARSE_BOOL_FALSE = 16,
   // PARSE_NULL = 17,
} K_PARSESTATE;

typedef enum
{
   CHAR_WHITESPACE = 0,
   CHAR_QUOTATION_MARK = 1,
   CHAR_ONENINE = 2,
   CHAR_LETTER = 3,
   CHAR_COLON = 4,
   CHAR_COMMA = 5,
   CHAR_OPEN_CURLY_BRACKET = 6,
   CHAR_CLOSE_CURLY_BRACKET = 7,
   CHAR_OPEN_SQUARE_BRACKET = 8,
   CHAR_CLOSE_SQUARE_BRACKET = 9,
   CHAR_ZERO = 10,
   CHAR_MINUS = 11,
   CHAR_PERIOD = 11,
   CHAR_E = 12,
   CHAR_PLUS = 13,
   CHAR_REVERSE_SOLIDUS = 14,
   CHAR_T = 15,
   CHAR_R = 16,
   CHAR_U = 17,
   CHAR_F = 18,
   CHAR_A = 19,
   CHAR_L = 20,
   CHAR_S = 21,
   CHAR_N = 22,
   CHAR_SOLIDUS = 23,
} K_CHARACTERTYPE;

typedef enum
{
   POINTER_TOKEN_INT = 0,
   POINTER_TOKEN_STR = 1,
} K_POINTERTOKENTYPE;

// Forward declaration.
struct json_value_s;

typedef struct json_array_s
{
   // The number of occupied elements in the array.
   unsigned int size;

   // The maximum number of occupied elements in the array.
   unsigned int capacity;

   // Dynamically allocated array of values.
   struct json_value_s * vals;
} json_array_t;

typedef struct json_object_s
{
   unsigned int size;

   unsigned int capacity;

   struct json_value_s * keys;

   struct json_value_s * values;
} json_object_t;

typedef struct json_string_s
{
   unsigned int size;

   unsigned int capacity;

   char * buffer;
} json_string_t;

typedef struct json_value_s
{
   K_VALUETYPE value_type;

   struct json_value_s * parent;

   union
   {
      json_object_t object;
      json_array_t array;
      json_string_t string;
      float fnum;
      int inum;
   };
} json_value_t;

typedef struct json_parse_stack_s
{
   int size;
   int capacity;
   K_PARSESTATE * vals;
} json_parse_stack_t;

typedef struct json_pointer_token_s
{
   union
   {
      int int_token;
      json_string_t str_token;
   };
} json_pointer_token_t;

#ifdef __cplusplus
}
#endif

#endif

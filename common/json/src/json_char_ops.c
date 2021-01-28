#include "json_char_ops.h"

#ifdef __cplusplus
extern "C"
{
#endif

K_CHARACTERTYPE json_character_type(int character)
{
   switch (character)
   {
      case ' ':
      case '\n':
      case '\r':
      case '\t':
      {
         return CHAR_WHITESPACE;
      }
      case '\\':
      {
         return CHAR_REVERSE_SOLIDUS;
      }
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      {
         return CHAR_ONENINE;
      }
      case '0':
      {
         return CHAR_ZERO;
      }
      case '-':
      {
         return CHAR_MINUS;
      }
      case ',':
      {
         return CHAR_COMMA;
      }
      case '+':
      {
         return CHAR_PLUS;
      }
      case '.':
      {
         return CHAR_PERIOD;
      }
      case 'e':
      case 'E':
      {
         return CHAR_E;
      }
      case 't':
      case 'T':
      {
         return CHAR_T;
      }
      case 'r':
      case 'R':
      {
         return CHAR_R;
      }
      case 'u':
      case 'U':
      {
         return CHAR_U;
      }
      case 'f':
      case 'F':
      {
         return CHAR_F;
      }
      case 'a':
      case 'A':
      {
         return CHAR_A;
      }
      case 'l':
      case 'L':
      {
         return CHAR_L;
      }
      case 's':
      case 'S':
      {
         return CHAR_S;
      }
      case 'n':
      case 'N':
      {
         return CHAR_N;
      }
      case '\"':
      {
         return CHAR_QUOTATION_MARK;
      }
      case ':':
      {
         return CHAR_COLON;
      }
      case '{':
      {
         return CHAR_OPEN_CURLY_BRACKET;
      }
      case '}':
      {
         return CHAR_CLOSE_CURLY_BRACKET;
      }
      case '[':
      {
         return CHAR_OPEN_SQUARE_BRACKET;
      }
      case ']':
      {
         return CHAR_CLOSE_SQUARE_BRACKET;
      }
      default:
         break;
   }

   return CHAR_LETTER;
}

#ifdef __cplusplus
}
#endif

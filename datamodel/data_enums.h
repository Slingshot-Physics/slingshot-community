#ifndef DATA_ENUMS_HEADER
#define DATA_ENUMS_HEADER

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
   DATA_CONSTRAINT_NONE = 0,
   DATA_CONSTRAINT_BALLJOINT = 1,
   DATA_CONSTRAINT_COLLISION = 2,
   DATA_CONSTRAINT_FRICTION = 3,
   DATA_CONSTRAINT_REVOLUTEJOINT = 4,
   DATA_CONSTRAINT_PISTON = 5,
   DATA_CONSTRAINT_REVOLUTEMOTOR = 6,
   DATA_CONSTRAINT_GEAR = 7,
   DATA_CONSTRAINT_TRANSLATION1D = 8,
   DATA_CONSTRAINT_ROTATION1D = 9,
   // Add em as you need em
} data_constraintType_t;

typedef enum
{
   DATA_SHAPE_CUBE = 0,
   DATA_SHAPE_CYLINDER = 1,
   DATA_SHAPE_SPHERE = 4,
   DATA_SHAPE_CAPSULE = 7,
   DATA_SHAPE_NONE = 404,
   // Add em as you need em
} data_shapeType_t;

typedef enum
{
   DATA_FRAME_BODY = 0,
   DATA_FRAME_GLOBAL = 1,
} data_frameType_t;

typedef enum
{
   DATA_NONE = 0,
   DATA_PROFILE = 1,
   // Add em as you need em
} data_loggingType_t;

#ifdef __cplusplus
}
#endif

#endif

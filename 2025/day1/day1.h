/* date = December 6th 2025 2:32 pm */

#ifndef DAY1_H
#define DAY1_H

//~ Types
typedef struct rotation rotation;
struct rotation
{
    b32 IsLeft;
    s32 Count;
};

typedef struct rotations_array rotations_array;
struct rotations_array
{
    rotation *Values;
    umm Count;
};

#endif //DAY1_H

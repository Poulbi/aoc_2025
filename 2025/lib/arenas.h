/* date = December 6th 2025 2:11 pm */

#ifndef ARENAS_H
#define ARENAS_H

typedef struct arena arena;
struct arena
{
    void *Base;
    umm Pos;
    umm Size;
};

#endif //ARENAS_H

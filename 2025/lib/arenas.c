#include "arenas.h"

arena *ArenaAlloc(void)
{
    umm DefaultSize = Kilobytes(64);
    
    arena *Arena = 0;
    
    void *Base = OS_Allocate(DefaultSize);
    
    Arena = (arena *)Base;
    Arena->Base = Base;
    Arena->Pos = sizeof(arena);
    Arena->Size = DefaultSize;
    
    return Arena;
}

void *ArenaPush(arena *Arena, umm Size)
{
    void *Result = Arena->Base + Arena->Pos;
    
    Assert(Arena->Pos + Size < Arena->Size);
    Arena->Pos += Size;
    
    return Result;
}

#define PushArray(Arena, type, Count) (type *)ArenaPush((Arena), (Count)*(sizeof(type)))
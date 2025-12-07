//~ Libarries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MemoryCopy memcpy

// Mine
#include "lr/lr.h"

//~ Layers
// OS
#include "os.h"
#include "arenas.c"
#include "lanes.c"

ENTRY_POINT(EntryPoint)
{
    // Thread init stuff
    {    
        ThreadContextSelect((thread_context *)Params);
        
        ThreadContext->Arena = ArenaAlloc();
        
        str8 ThreadName = {0};
        ThreadName.Data = (u8[16]){0};
        ThreadName.Size = 1;
        ThreadName.Data[0] = (u8)LaneIndex() + '0';
        OS_SetThreadName(ThreadName);
    }
    
    return 0;
}
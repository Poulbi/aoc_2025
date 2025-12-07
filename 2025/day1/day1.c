//~ Libarries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Mine
#include "lr/lr.h"

#define MemoryCopy memcpy

//~ Layers
// OS
#include "day1.h"
#include "os.h"
#include "arenas.c"
#include "lanes.c"

#define MAX 100

//~ Main program
s32 RotateAndIncrementPasswordPartOne(s32 *ArrowPos, rotation Rotation)
{
    s32 Result = 0;
    s32 Arrow = *ArrowPos;
    
    Rotation.Count %= MAX;
    
    // Rotate the cursor
    if(Rotation.IsLeft)
    {
        Arrow = ((MAX+Arrow) - Rotation.Count)%MAX;
    }
    else
    {
        Arrow = (Arrow + Rotation.Count)%MAX;
    }
    Assert(Arrow >= 0 && Arrow < MAX);
    
    Result = (Arrow == 0);
    
    *ArrowPos = Arrow;
    
    return Result;
}

s32 RotateAndIncrementPasswordPartTwo(s32 *ArrowPos, rotation Rotation)
{
    s32 Result = 0;
    s32 Arrow = *ArrowPos;
    
    s32 Turns = Rotation.Count / MAX; 
    Rotation.Count %= MAX;
    
    if(Arrow != 0)
    {
        if(Rotation.IsLeft && (Rotation.Count > Arrow))
        {
            Turns += 1;
        }
        else if(!Rotation.IsLeft && (Arrow + Rotation.Count > MAX))
        {
            Turns += 1;
        }
    }
    
    //Rotate the arrow
    if(Rotation.IsLeft)
    {
        Arrow = ((MAX+Arrow) - Rotation.Count)%MAX;
    }
    else
    {
        Arrow = (Arrow + Rotation.Count)%MAX;
    }
    Assert(Arrow >= 0 && Arrow < MAX);
    
    Result = Turns + !!(Arrow == 0);
    
    *ArrowPos = Arrow;
    
    return Result;
}

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
    
    s64 ValuesCount = 0;
    s64 *Values = 0;
    
    str8 FileBuffer = {0};
    str8 *File = 0;
    rotations_array *RotationsTable = 0;
    
    // TODO(luca): Multi-threaded read of the same file? 
    if(LaneIndex() == 0)
    {
        FileBuffer = OS_ReadEntireFileIntoMemory("./2025/day1/input");
        File = &FileBuffer;
        if(!File->Size)
        {
            fprintf(stderr, "ERROR: Could not read file.\n");
        }
        RotationsTable = PushArray(ThreadContext->Arena, rotations_array, (LaneCount()));
    }
    LaneSyncU64((u64 *)&File, 0);
    LaneSyncU64((u64 *)&RotationsTable, 0);
    
    range_s64 Range = LaneRange(File->Size);
    rotations_array Rotations = {0};
    // Allocate the maximum amount of possible rotations (e.g., a file containing "L123...789\nEOF")
    umm MaxRotationsCount = (Range.Max - Range.Min - 2);
    Rotations.Values = PushArray(ThreadContext->Arena, rotation, MaxRotationsCount);
    
    // Parse the file 
    
    u8 *In = File->Data;
    for(s64 At = Range.Min; At < Range.Max; At += 1)
    {
        b32 RotateLeft = false;
        b32 RotateCount = 0;
        
        // NOTE(luca): If the character at the start of the range is not an R or an L we should skip until we find an R or an L.  This will account for when a range does not stop at a newline.
        {
            while(At >= 0 && (In[At] != 'R' && In[At] != 'L') )
            {
                At -= 1;
            }
            Assert(At >= 0);
        }
        
        RotateLeft = (In[At] == 'R');
        At += 1;
        
        while(At < Range.Max && In[At] != '\n')
        {
            s32 Digit = (In[At] - '0');
            RotateCount = 10*RotateCount + Digit;
            At += 1;
        }
        
        b32 IsLastRange = (LaneIndex() == (LaneCount() - 1));
        if(At < Range.Max || IsLastRange)
        {
            rotation *Rotation = Rotations.Values + Rotations.Count;
            Rotation->IsLeft = RotateLeft;
            Rotation->Count  = RotateCount;
            Rotations.Count  += 1;
        }
    }
    
    RotationsTable[LaneIndex()] = Rotations;
    LaneSync();
    
    if(LaneIndex() == 0)
    {
        s32 Arrow = 50;
        s32 Password = 0;
        
        for(EachIndex(Index, LaneCount()))
        {
            rotations_array Rotations = RotationsTable[Index];
            for(EachIndex(Index, Rotations.Count))
            {
                rotation Rotation = Rotations.Values[Index];
                
#if 0                
                printf("%c%d\n", ((Rotation.IsLeft) ? 'L' : 'R'), Rotation.Count);
#endif
                
                Password += RotateAndIncrementPasswordPartTwo(&Arrow, Rotation);
            }
        }
        
        printf("Password is %d.\n", Password);
    }
    
    return 0;
}
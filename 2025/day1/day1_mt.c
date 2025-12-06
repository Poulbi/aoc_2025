//~ Libarries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Linux
#include <pthread.h>
#include <linux/prctl.h> 
#include <sys/prctl.h>
// Mine
#include "lr/lr.h"
#include "os.c"

#define MemoryCopy memcpy

//~ Types
typedef u64 barrier;

typedef struct lane_context lane_context;
struct lane_context
{
    s64 LaneCount;
    s64 LaneIndex;
    
    u64 *SharedStorage;
    barrier Barrier;
};

//~ Globals
global s64 GlobalTotalSum;
thread_static lane_context *ThreadContext;

//~ Layers
#include "linux.c"
#include "lanes.c"

//~ Macro's
#define AtomicAdd64(Value, Add) (__atomic_fetch_add((u64 *)(Value), (Add), __ATOMIC_SEQ_CST) + (Add))

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


#define MAX 100

s32 RotateAndIncrementPasswordPartOne(s32 *CursorPos, rotation Rotation)
{
    s32 Result = 0;
    s32 Cursor = *CursorPos;
    
    Rotation.Count %= MAX;
    
    // Rotate the cursor
    if(Rotation.IsLeft)
    {
        Cursor = ((MAX+Cursor) - Rotation.Count)%MAX;
    }
    else
    {
        Cursor = (Cursor + Rotation.Count)%MAX;
    }
    Assert(Cursor >= 0 && Cursor < MAX);
    
    Result = (Cursor == 0);
    
    *CursorPos = Cursor;
    
    return Result;
}

s32 RotateAndIncrementPasswordPartTwo(s32 *CursorPos, rotation Rotation)
{
    s32 Result = 0;
    s32 Cursor = *CursorPos;
    
    s32 Turns = Rotation.Count / MAX; 
    Rotation.Count %= MAX;
    
    if(Cursor != 0)
    {
        if(Rotation.IsLeft && (Rotation.Count > Cursor))
        {
            Turns += 1;
        }
        else if(!Rotation.IsLeft && (Cursor + Rotation.Count > MAX))
        {
            Turns += 1;
        }
    }
    
    //Rotate the cursor
    if(Rotation.IsLeft)
    {
        Cursor = ((MAX+Cursor) - Rotation.Count)%MAX;
    }
    else
    {
        Cursor = (Cursor + Rotation.Count)%MAX;
    }
    Assert(Cursor >= 0 && Cursor < MAX);
    
    Result = Turns + !!(Cursor == 0);
    
    *CursorPos = Cursor;
    
    return Result;
}


//~ Main
void *EntryPoint(void *Params)
{
    // Thread init stuff
    {    
        ThreadContext = (lane_context *)Params;
        
        str8 ThreadName = {0};
        ThreadName.Data = (u8[16]){0};
        ThreadName.Size = 1;
        ThreadName.Data[0] = (u8)LaneIndex() + '0';
        OS_SetThreadName(ThreadName);
    }
    
    s64 ValuesCount = 0;
    s64 *Values = 0;
    
    str8 *File = 0;
    
#if 0    
    str8 FileBuffer = ReadEntireFileIntoMemory("./input_short");
    File = &FileBuffer;
#else
    // TODO(luca): Multi-threaded read of the same file? 
    if(LaneIndex() == 0)
    {
        File = malloc(sizeof(*File));
        *File = ReadEntireFileIntoMemory("./2025/day1/input");
        if(!File->Size)
        {
            fprintf(stderr, "ERROR: Could not read file.\n");
        }
    }
    LaneSyncU64((u64 *)&File, 0);
#endif
    
    range_s64 Range = LaneRange(File->Size);
    
    // 1. Parse the whole file 
    // NOTE(luca): if the character is not an R or an L we should skip until we find an R or an L.  This will account for when a range does not stop at a newline.
    
    rotations_array Rotations = {0};
    // Allocate the maximum amount of possible rotations (e.g., a file containing "L123...789\nEOF")
    umm MaxRotationsCount = (Range.Max - Range.Min - 2);
    Rotations.Values = (rotation *)malloc((MaxRotationsCount)*sizeof(rotation));
    
    u8 *In = File->Data;
    for(s64 At = Range.Min; At < Range.Max; At += 1)
    {
        b32 RotateLeft = false;
        b32 RotateCount = 0;
        
        // First
        while(At >= 0 && (In[At] != 'R' && In[At] != 'L') )
        {
            At -= 1;
        }
        Assert(At >= 0);
        
        if(In[At] == 'R')
        {
            RotateLeft = false;
        }
        else if(In[At] == 'L')
        {
            RotateLeft = true;
        }
        else
        {
            Assert(0);
        }
        At += 1;
        
        // NOTE(luca): This should take care of the second part 
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
#if 0            
            // NOTE(luca): To validate, you can use this printout against the input file, just be sure to sort both.
            printf("%c%d\n", ((RotateLeft) ? 'L' : 'R'), RotateCount);
#endif
        }
    }
    
    rotations_array *RotationsTable = 0;
    if(LaneIndex() == 0)
    {
        RotationsTable = (rotations_array *)malloc(LaneCount()*sizeof(rotations_array));
    }
    LaneSyncU64((u64 *)&RotationsTable, 0);
    
    RotationsTable[LaneIndex()] = Rotations;
    
    LaneSync();
    
    if(LaneIndex() == 0)
    {
        s32 Cursor = 50;
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
                
                Password += RotateAndIncrementPasswordPartOne(&Cursor, Rotation);
            }
        }
        
        printf("Password is %d.\n", Password);
    }
    
    // TODO(luca): 
    // 2. On a single lane go over each rotation and calculate the password.
    
    return 0;
}
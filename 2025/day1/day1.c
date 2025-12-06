#include "lr/lr.h"
#include "linux.c"

#define MAX_NUMBER 99


int main(int ArgCount, char *Args[])
{
    if(ArgCount >= 2)
    { 
        u32 Password = 0;
        str8 In = OS_ReadEntireFileIntoMemory(Args[1]);
        
        // Where the arrow is pointing at
        s32 Cursor = 50; 
        
        for(umm At = 0; At < In.Size; At += 1)
        {
            b32 RotateLeft = false;
            s32 RotateCount = 0;
            
            // Parse rotation
            {
                if(In.Data[At] == 'R')
                {
                    RotateLeft = false;
                }
                else if(In.Data[At] == 'L')
                {
                    RotateLeft = true;
                }
                else
                {
                    Assert(0);
                }
                
                At += 1;
                
                while(At < In.Size && In.Data[At] != '\n')
                {
                    s32 Digit = (In.Data[At] - '0'); 
                    RotateCount = 10*RotateCount + Digit;
                    At += 1;
                }
                Assert(In.Data[At] == '\n');
                
#if 0                
                LogFormat("%c%lu\n", ((RotateLeft) ? 'L' : 'R'), RotateCount);
#endif
                
            }
            
#define MAX 100
            
#if 0 // Method for part one
            RotateCount %= MAX;
            
            // Rotate the cursor
            if(RotateLeft)
            {
                Cursor = ((MAX+Cursor) - RotateCount)%MAX;
            }
            else
            {
                Cursor = (Cursor + RotateCount)%MAX;
            }
            Assert(Cursor >= 0 && Cursor < MAX);
            
            if(Cursor == 0)
            {
                Password += 1;
            }
            
#elif 1 // Method for part two
            s32 Rotations = RotateCount / MAX; 
            
            RotateCount %= MAX;
            
            if(Cursor != 0)
            {
                if(RotateLeft && (RotateCount > Cursor))
                {
                    Rotations += 1;
                }
                else if(!RotateLeft && (Cursor + RotateCount > MAX))
                {
                    Rotations += 1;
                }
            }
            
            
            //Rotate the cursor
            if(RotateLeft)
            {
                Cursor = ((MAX+Cursor) - RotateCount)%MAX;
            }
            else
            {
                Cursor = (Cursor + RotateCount)%MAX;
            }
            Assert(Cursor >= 0 && Cursor < MAX);
            
            if(Cursor == 0)
            {
                Password += 1;
            }
            Password += Rotations;
#endif
            
        }
        
        OS_PrintFormat("Password is %llu.\n", Password);
    }
    else
    {
        OS_PrintFormat("ERROR: No input file specified.\n");
        OS_PrintFormat("Usage: %s <input>\n", Args[0]);
    }
    
    return 0;
}
#include "base/base.h"

#define LaneSingleOSPrint(Format, ...) if(LaneIndex() == 0) { OS_PrintFormat(Format, ##__VA_ARGS__); }

#define BatteryCount 12

s32 GetJoltageSumPartOne(str8 InputFile, range_s64 InputRange)
{
    s32 JoltageSum = 0;
    
    u8 *In = InputFile.Data;
    s64 At = InputRange.Min;
    
    // Get At to be on the start of a bank
    // This if statement has a nice side effect that if ranges overlap (i.e., two ranges go over the same bank), only the last one that includes the end will be used since the while loop below ends at `InputRange.Max` . 
    if(At > 0)
    {
        while(At >= 0 && In[At] != '\n')
        {
            At -= 1;
        }
        At += 1;
    }
    
    
    while(At < InputRange.Max)
    {
        s64 Start = At;
        s32 First  = (In[At + 0] - '0');
        s32 Second = (In[At + 1] - '0');
        At += 2;
        
        while(At < InputRange.Max && In[At] != '\n')
        {
            s32 Digit = (s32)(In[At] - '0'); 
            
            if(Second > First)
            {
                First = Second;
                Second = Digit;
            }
            else if(Digit > Second)
            {
                Second = Digit;
            }
            else
            {
                // Skip this digit
            }
            
            At += 1;
        }
        
        if(At == InputFile.Size || In[At] == '\n')
        {
            
#if 0            
            OS_PrintFormat("%.*s %d%d\n",
                           (s32)(At - Start), (In + Start), First, Second);
#endif
            
            JoltageSum += 10*First + Second;
            At += 1;
        }
    }
    
    return JoltageSum;
}

ENTRY_POINT(EntryPoint)
{
    ThreadInit((thread_context *)Params);
    
    str8 InputFile = {0};
    range_s64 InputRange = {0};
    
    if(Params->ArgsCount >= 2)
        
    {        
        InputFile = OS_ReadEntireFileIntoMemory(Params->Args[1]);
        
        // TODO(luca): Create ranges based on finding a newline.
        
        InputRange = LaneRange(InputFile.Size);
        
        if(InputFile.Size)
        {        
            s32 JoltageSum = GetJoltageSumPartOne(InputFile, InputRange);
            
            s64 TotalJoltageSum = 0;
            s64 *TotalJoltageSumPtr = &TotalJoltageSum;
            
            LaneSyncU64((u64 *)&TotalJoltageSumPtr, 0);
            AtomicAddEvalU64(TotalJoltageSumPtr, JoltageSum);
            LaneIceberg();
            
            LaneSingleOSPrint("Joltage sum is %d.\n", *TotalJoltageSumPtr);
        }
        else
        {
            LaneSingleOSPrint("ERROR: Could not read file.\n");
        }
    }
    else
    {
        LaneSingleOSPrint("ERROR: No input provided."
                          "Usage: %s <input>\n", Params->Args[0]);
    }
    
    return 0;
}
#include "base/base.h"

#define LaneSingleOSPrint(Format, ...) if(LaneIndex() == 0) { OS_PrintFormat(Format, ##__VA_ARGS__); }

s64 GetInvalIDsSumForRangePartOne(range_s64 IDRange)
{
    s64 InvalidIDsSum = 0;
    
    // NOTE(luca): We can cache a lot of these calculations
    // - the digit count mostly stays the same
    // - the factors for the digit count as well
    for(s64 ID = IDRange.Min; ID <= IDRange.Max; ID += 1)
    {
        s64 DigitsCount = 0;
        
        // Count digits in ID
        {                
            s64 Value = ID;
            while(Value)
            {
                Value /= 10;
                DigitsCount += 1;
            }
        }
        
        for(u64 Divider = 1; Divider <= DigitsCount/2; Divider += 1)
        {
            if(DigitsCount%Divider == 0)
            {
                // This ID can be invalid
                s64 GroupSize = Divider;
                s64 GroupRepeatCount = DigitsCount/Divider;
                
                // Only do group repeats of 2
                if(GroupRepeatCount%2 == 0)
                {
                    s64 Shift = 10;
                    for(s64 Index = 1; Index < GroupSize; Index += 1) Shift *= 10;
                    
                    s64 GroupValue = ID%Shift;
                    
                    b32 IDIsValid = false;
                    s64 TempID = ID;
                    for(s64 Index = 0; Index < GroupRepeatCount; Index += 1)
                    {
                        s64 Group = TempID%Shift;
                        if(GroupValue != Group)
                        {
                            IDIsValid = true;
                            break;
                        }
                        TempID /= Shift;
                    }
                    
                    if(!IDIsValid)
                    {
                        
#if 0                                
                        OS_PrintFormat("[%ld] Invalid ID: %ld\n", LaneIndex(), ID);
#endif
                        
                        InvalidIDsSum += ID;
                        break;
                    }
                    
                }
            }
        }
        
    }
    
    return InvalidIDsSum;
}


s64 GetInvalIDsSumForRangePartTwo(range_s64 IDRange)
{
    s64 InvalidIDsSum = 0;
    
    // NOTE(luca): We can cache a lot of these calculations
    // - the digit count mostly stays the same
    // - the factors for the digit count as well
    for(s64 ID = IDRange.Min; ID <= IDRange.Max; ID += 1)
    {
        s64 DigitsCount = 0;
        
        // Count digits in ID
        {                
            s64 Value = ID;
            while(Value)
            {
                Value /= 10;
                DigitsCount += 1;
            }
        }
        
        for(u64 Divider = 1; Divider <= DigitsCount/2; Divider += 1)
        {
            if(DigitsCount%Divider == 0)
            {
                // This ID can be invalid
                s64 GroupSize = Divider;
                s64 GroupRepeatCount = DigitsCount/Divider;
                
                s64 Shift = 10;
                for(s64 Index = 1; Index < GroupSize; Index += 1) Shift *= 10;
                
                s64 GroupValue = ID%Shift;
                
                b32 IDIsValid = false;
                s64 TempID = ID;
                for(s64 Index = 0; Index < GroupRepeatCount; Index += 1)
                {
                    s64 Group = TempID%Shift;
                    if(GroupValue != Group)
                    {
                        IDIsValid = true;
                        break;
                    }
                    TempID /= Shift;
                }
                
                if(!IDIsValid)
                {
                    
#if 0                                
                    OS_PrintFormat("[%ld] Invalid ID: %ld\n", LaneIndex(), ID);
#endif
                    
                    InvalidIDsSum += ID;
                    break;
                }
                
            }
        }
    }
    
    return InvalidIDsSum;
}


ENTRY_POINT(EntryPoint)
{
    s64 InvalidIDsSum = 0;
    str8 InputFile = {0};
    range_s64 InputRange = {0};
    
    if(Params->ArgsCount >= 2)
    {
        InputFile = OS_ReadEntireFileIntoMemory(Params->Args[1]);
        if(InputFile.Size)
        {    
            InputRange = LaneRange(InputFile.Size);
            
            LaneSingleOSPrint("Input size: %lu\n", InputFile.Size);
            
            // 1. Parse range
            // 2. Check for invalid IDs
            // 3. Sum all invalid IDs
            // 4. Aggregate
            {    
                u8 *In = InputFile.Data;
                s64 At = InputRange.Min;
                
                // Walk back to previous comma if range does not start on one
                if(At != 0)
                {
                    while(At >= 0 && In[At] != ',') At -= 1;
                    if(At < 0)
                    {
                        OS_PrintFormat("ERROR(%d): Expected ',' found BOF\n", InputRange.Min);
                        return 0;
                    }
                    At += 1;
                }
                
                for(; At < InputRange.Max; At += 1)
                {
                    range_s64 IDRange = {0};
                    
                    s64 ParsedValue = 0;
                    
                    while(In[At] != '-')
                    {
                        Assert(In[At] >= '0' && In[At] <= '9');
                        s64 Digit = (In[At] - '0');
                        ParsedValue = 10*ParsedValue + Digit; 
                        At += 1;
                    }
                    IDRange.Min = ParsedValue;
                    
                    At += 1;
                    
                    ParsedValue = 0;
                    while(At < InputRange.Max && In[At] != ',')
                    {
                        Assert(In[At] >= '0' && In[At] <= '9');
                        s64 Digit = (In[At] - '0');
                        ParsedValue = 10*ParsedValue + Digit; 
                        
                        At += 1;
                    }
                    IDRange.Max = ParsedValue;
                    
                    if(In[At] == ',' || At == InputFile.Size)
                    {
#if 1
                        // To validate
                        // TODO(luca): thread-safe OS_PrintFormat ?
                        printf("[%ld] %ld-%ld\n", LaneIndex(), IDRange.Min, IDRange.Max);
#endif
                        
                        InvalidIDsSum += GetInvalIDsSumForRangePartTwo(IDRange);
                    }
                }
            }
            
#if 0    
            OS_PrintFormat("%ld: Sum: %ld\n", LaneIndex(), InvalidIDsSum);
#endif
            
            s64 *Sums = 0;
            if(LaneIndex() == 0)
            {
                Sums = PushArray(ThreadContext->Arena, s64, LaneCount()); 
            }
            LaneSyncU64((u64 *)&Sums, 0);
            
            Sums[LaneIndex()] = InvalidIDsSum;
            LaneIceberg();
            
            s64 TotalSum = 0;
            for(EachIndex(Index, LaneCount()))
            {
                TotalSum += Sums[Index];
            }
            
            LaneSingleOSPrint("Invalid IDs sum: %ld\n", TotalSum);
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

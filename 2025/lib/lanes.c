#include "lanes.h"

thread_static thread_context *ThreadContext;

#define LaneCount() (ThreadContext->LaneCount)
#define LaneIndex() (ThreadContext->LaneIndex)

void ThreadContextSelect(thread_context *Context)
{
    ThreadContext = Context;
}

void LaneSync(void)
{
    OS_BarrierWait(ThreadContext->Barrier);
}

void LaneSyncU64(u64 *Value, s64 SourceIndex)
{
    if(LaneIndex() == SourceIndex)
    {
        MemoryCopy(ThreadContext->SharedStorage, Value, sizeof(u64));
    }
    LaneSync();
    
    if(LaneIndex() != SourceIndex)
    {
        MemoryCopy(Value, ThreadContext->SharedStorage, sizeof(u64));
    }
    LaneSync();
}

range_s64 LaneRange(s64 ValuesCount)
{
    range_s64 Result = {0};
    
    s64 ValuesPerThread = ValuesCount/LaneCount();
    
    s64 LeftoverValuesCount = ValuesCount%LaneCount();
    b32 ThreadHasLeftover = (LaneIndex() < LeftoverValuesCount);
    s64 LeftoversBeforeThisThreadIndex = ((ThreadHasLeftover) ? 
                                          LaneIndex(): 
                                          LeftoverValuesCount);
    
    Result.Min = (ValuesPerThread*LaneIndex()+
                  LeftoversBeforeThisThreadIndex);
    Result.Max = (Result.Min + ValuesPerThread + !!ThreadHasLeftover);
    
    return Result;
}

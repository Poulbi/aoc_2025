
//~ Defined previously
void *EntryPoint(void *Params);

//~ Types
typedef struct os_thread os_thread;
struct os_thread
{
    pthread_t Handle;
    void *Result;
    
    lane_context Context;
};

//~ Implementation
void OS_BarrierWait(barrier Barrier)
{
    pthread_barrier_wait((pthread_barrier_t *)Barrier);
}

void OS_SetThreadName(str8 ThreadName)
{
    Assert(ThreadName.Size <= 16 -1);
    prctl(PR_SET_NAME, ThreadName);
}

void LinuxMainEntryPoint(int ArgsCount, char **Args)
{
    char ThreadName[16] = "Main";
    os_thread Threads[2/2] = {0};
    s32 Ret = 0;
    s64 ThreadsCount = 2/2;
    
    prctl(PR_SET_NAME, ThreadName);
    
    u64 SharedStorage = 0;
    pthread_barrier_t Barrier;
    pthread_barrier_init(&Barrier, 0, (u32)ThreadsCount);
    
    for(s64 Index = 0; Index < ThreadsCount; Index += 1)
    {
        Threads[Index].Context.LaneIndex = Index;
        Threads[Index].Context.LaneCount = ThreadsCount;
        Threads[Index].Context.Barrier   = (barrier)(&Barrier);
        Threads[Index].Context.SharedStorage = &SharedStorage;
        
        Ret = pthread_create(&Threads[Index].Handle, 0, EntryPoint, &Threads[Index].Context);
        Assert(Ret == 0);
    }
    
    for(s64 Index = 0; Index < ThreadsCount; Index += 1)
    {
        pthread_join(Threads[Index].Handle, &Threads[Index].Result);
    }
}

int main(int ArgsCount, char **Args)
{
    LinuxMainEntryPoint(ArgsCount, Args);
    return 0;
}
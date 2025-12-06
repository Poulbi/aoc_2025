// Linux
#include <pthread.h>
#include <linux/prctl.h> 
#include <sys/prctl.h>

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define STB_SPRINTF_IMPLEMENTATION
#include "stb_sprintf.h"

#include "lanes.h"
#include "linux.h"

global u8 LogBuffer[Kilobytes(64)];

#if !defined(NUMBER_OF_CORES)
# define NUMBER_OF_CORES 6
#endif

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

//~ Syscalls

//- Debug utilities 
void AssertErrnoNotEquals(smm Result, smm ErrorValue)
{
    if(Result == ErrorValue)
    {
        int Errno = errno;
        Assert(0);
    }
}

void AssertErrnoEquals(smm Result, smm ErrorValue)
{
    if(Result != ErrorValue)
    {
        int Errno = errno;
        Assert(0);
    }
}

str8 OS_ReadEntireFileIntoMemory(char *FileName)
{
    str8 Result = {};
    
    if(FileName)
    {
        int File = open(FileName, O_RDONLY);
        
        if(File != -1)
        {
            struct stat StatBuffer = {};
            int Error = fstat(File, &StatBuffer);
            AssertErrnoNotEquals(Error, -1);
            
            Result.Size = StatBuffer.st_size;
            Result.Data = (u8 *)mmap(0, Result.Size, PROT_READ, MAP_PRIVATE, File, 0);
            AssertErrnoNotEquals((smm)Result.Data, (smm)MAP_FAILED);
        }
    }
    
    return Result;
}

void OS_PrintFormat(char *Format, ...)
{
    va_list Args;
    va_start(Args, Format);
    
    int Length = stbsp_vsprintf((char *)LogBuffer, Format, Args);
    smm BytesWritten = write(STDOUT_FILENO, LogBuffer, Length);
    AssertErrnoEquals(BytesWritten, Length);
}

//~ Threads
void OS_BarrierWait(barrier Barrier)
{
    pthread_barrier_wait((pthread_barrier_t *)Barrier);
}

void OS_SetThreadName(str8 ThreadName)
{
    Assert(ThreadName.Size <= 16 -1);
    prctl(PR_SET_NAME, ThreadName);
}

#ifdef LINUX_LANE_ENTRYPOINT

//~ Entrypoint
void LinuxMainEntryPoint(int ArgsCount, char **Args)
{
    char ThreadName[16] = "Main";
    os_thread Threads[NUMBER_OF_CORES] = {0};
    s32 Ret = 0;
    s64 ThreadsCount = NUMBER_OF_CORES;
    
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

#endif // LINUX_LANE_ENTRYPOINT
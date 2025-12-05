
#include <stdio.h>

#include <semaphore.h>

/*
sem_init

sem_post
sem_wait
*/

#include <pthread.h>

#include "lr/lr.h"

#define NUMBER_OF_CORES 4

void *EntryPoint(void *Params);

struct thread
{
    pthread_t Thread;
    void *Result;
};
typedef struct thread thread;

global pthread_barrier_t GlobalThreadBarrier;

int main(void)
{
    thread Threads[NUMBER_OF_CORES] = {0};
    
    pthread_barrier_init(&GlobalThreadBarrier, 0, NUMBER_OF_CORES);
    
    for(s64 ThreadIndex = 0; ThreadIndex < NUMBER_OF_CORES; ThreadIndex += 1)
    {
        s32 Result = pthread_create(&Threads[ThreadIndex].Thread, 0, EntryPoint, (void *)ThreadIndex);
        Assert(Result == 0);
    }
    
    for(s64 ThreadIndex = 0; ThreadIndex < NUMBER_OF_CORES; ThreadIndex += 1)
    {
        pthread_join(Threads[ThreadIndex].Thread, &Threads[ThreadIndex].Result);
    }
    
    return 0;
}

void BarrierSync(void)
{
    pthread_barrier_wait(&GlobalThreadBarrier);
}

global s64 ThreadSums[NUMBER_OF_CORES]; 

void *EntryPoint(void *Params)
{
    s64 ThreadIndex = (s64)Params;
    s64 ThreadCount = NUMBER_OF_CORES;
    
    s64 ValuesCount = 129;
    s64 *Values = 0;
    s64 ThreadSum = 0;
    s64 ValuesPerThread = ValuesCount/ThreadCount;
    
    s64 LeftoverValuesCount = ValuesCount%ThreadCount;
    b32 ThreadHasLeftover = (ThreadIndex < LeftoverValuesCount);
    s64 LeftoversBeforeThisThreadIndex = ((ThreadHasLeftover) ? 
                                          ThreadIndex : 
                                          LeftoverValuesCount);
    
    s64 ThreadFirstValueIndex = (ValuesPerThread*ThreadIndex +
                                 LeftoversBeforeThisThreadIndex);
    s64 ThreadOnePastLastValueIndex = (ThreadFirstValueIndex + ValuesPerThread + !!ThreadHasLeftover);
    
    ThreadSums[ThreadIndex] = ThreadSum;
    printf("%ld\n", ThreadIndex);
    // Sync
    //BarrierSync();
    printf("%ld\n", ThreadIndex);
    
    s64 Sum = 0;
    for(s64 ThreadIndex = 0; ThreadIndex < NUMBER_OF_CORES; ThreadIndex += 1)
    {
        Sum += ThreadSums[ThreadIndex];
    }
    
    printf("Index: %ld, Values: %ld-%ld, Leftovers: %ld\n", 
           ThreadIndex,
           ThreadFirstValueIndex, ThreadOnePastLastValueIndex,
           LeftoversBeforeThisThreadIndex);
    
    
    return 0;
}
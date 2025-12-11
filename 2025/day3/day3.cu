#include <cuda_runtime.h>

#include "base/base.h"

#define BatteryCount 12

#if AOC_INTERNAL
# if defined(Assert)
#  undef Assert
# endif
# define Assert(Expression)
#endif

#if AOC_INTERNAL
# define GPU_Assert(Expression) if(!(Expression)) { *(int *)0 = 0; }
#else
# define GPU_Assert(Expression)
#endif

#if AOC_INTERNAL
# define CU_Check(Expression) { CU_Check_((Expression), __FILE__, __LINE__); }
#else
# define CU_Check(Expression) Expression
#endif
inline void CU_Check_(cudaError_t Code, char *FileName, s32 Line, b32 Abort=false)
{
    if(Code != cudaSuccess) 
    {
        fprintf(stderr,"ERROR: %s %s %d\n", cudaGetErrorString(Code), FileName, Line);
        if(Abort) 
        {
            exit(Code);
        }
    }
}

arena *CU_ArenaAlloc(arena *CPUArena)
{
    umm DefaultSize = Kilobytes(64);
    
    arena *Arena = PushStruct(CPUArena, arena);
    
    void *Base = 0;
    CU_Check(cudaMalloc(&Base, DefaultSize));
    
    Arena->Base = Base;
    Arena->Pos = 0;
    Arena->Size = DefaultSize;
    
    return Arena;
}

__global__ void
GetBigJoltage(u64 *Output, u8 *Lines, s32 LinesCount, s32 LineSize)
{
    u8 Idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(Idx < LinesCount)
    {
        u8 *In = Lines + LineSize*Idx;
        u64 *Out = Output + Idx;
        
        u64 Number = 0;
        s32 LastFoundPos = 0;
        for(s32 BatteryIdx = 0; BatteryIdx < BatteryCount; BatteryIdx += 1)
        {
            s32 Start = LastFoundPos;
            s32 End = LineSize - (BatteryCount - BatteryIdx);
            
            // NOTE(luca): There are no zeroes in the input.
            u8 Digit = '0';
            for(s32 At = Start; At < End; At += 1)
            {
                GPU_Assert(In[At] >= '0' && In[At] <= '9');
                if(In[At] > Digit)
                {
                    Digit = In[At];
                    LastFoundPos = At + 1;
                }
            }
            
            Number = Number*10 + (u64)(Digit - '0');
        }
        
        *Out = Number;
    }
}

ENTRY_POINT(EntryPoint)
{
    if(LaneIndex() == 0)
    {
        if(Params->ArgsCount >= 2)
        {            
            str8 File = OS_ReadEntireFileIntoMemory(Params->Args[1]);
            if(File.Size)
                
            {                
                umm LineSize = 0;
                for(umm Idx = 0; Idx < File.Size; Idx += 1)
                {
                    if(File.Data[Idx] == '\n')
                    {
                        LineSize = Idx + 1;
                        break;
                    }
                }
                Assert(LineSize < File.Size);
                Assert(File.Size%LineSize == 0);
                
                s32 LinesCount = (s32)(File.Size / LineSize); 
                s32 ThreadsPerBlock = 32;
                s32 BlocksCount = (LinesCount/ThreadsPerBlock) + 1;
                
                umm OutputSize = (sizeof(u64)*BatteryCount*LinesCount);
                u64 *HostOutput = (u64 *)ArenaPush(ThreadContext->Arena, OutputSize);
                
                // NOTE(luca): Trigger CUDA context initialization
                CU_Check(cudaSetDevice(0));
                
                // Get occupancy metrics
                {                
                    s32 IntendedSharedMemorySize = 0;
                    s32 MaxBlockSize = 0;
                    s32 MinGridSize;
                    s32 MinBlockSize;
                    CU_Check(cudaOccupancyMaxPotentialBlockSize(&MinGridSize, &MinBlockSize, GetBigJoltage, IntendedSharedMemorySize, MaxBlockSize));
                    
                    s32 BlocksCount;
                    CU_Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&BlocksCount, GetBigJoltage, MinGridSize, IntendedSharedMemorySize));
                    NullExpression;
                }
                
                // Run GPU code
                {                
                    arena *DeviceArena = CU_ArenaAlloc(ThreadContext->Arena);
                    u8  *DeviceLines  = PushArray(DeviceArena, u8,  File.Size);
                    u64 *DeviceOutput = PushArray(DeviceArena, u64, LinesCount);
                    
                    CU_Check(cudaMemcpy(DeviceLines, File.Data, File.Size, cudaMemcpyHostToDevice));
                    
                    cudaEvent_t Start, Stop;
                    CU_Check(cudaEventCreate(&Start));
                    CU_Check(cudaEventCreate(&Stop));
                    
                    CU_Check(cudaEventRecord(Start, 0));
                    
                    GetBigJoltage<<<BlocksCount, ThreadsPerBlock>>>(DeviceOutput, DeviceLines, LinesCount, (s32)LineSize);
                    
                    CU_Check(cudaEventRecord(Stop, 0));
                    CU_Check(cudaEventSynchronize(Stop));
                    
                    f32 ElapsedMS = 0;
                    CU_Check(cudaEventElapsedTime(&ElapsedMS, Start, Stop));
                    OS_PrintFormat("Elapsed time: %fms\n", (double)ElapsedMS);
                    
                    CU_Check(cudaGetLastError()); 
                    
                    CU_Check(cudaDeviceSynchronize());
                    CU_Check(cudaMemcpy(HostOutput, DeviceOutput, OutputSize, cudaMemcpyDeviceToHost));
                }
                
                // Aggregate results
                {
                    u64 JoltageSum = 0;
                    for(s32 Idx = 0; Idx < LinesCount; Idx += 1)
                    {
                        JoltageSum += HostOutput[Idx];
#if 0
                        OS_PrintFormat("%lu\n", HostOutput[Idx]);
#endif
                    }
                    OS_PrintFormat("Joltage sum is %lu.\n", JoltageSum);
                }
                
            }
            else
            {
                OS_PrintFormat("ERROR: Could not read file.\n");
            }
        }
        else
        {
            OS_PrintFormat("ERROR: No input provided.\n"
                           "Usage: %s <input>\n", Params->Args[0]);
        }
    }
    
    return 0;
}
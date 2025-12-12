#define FORCE_THREADS_COUNT 1
#include "base/base.h"

#include "cu.h"

global_variable void (*Log)(char *Format, ...) = OS_PrintFormat;

kernel void
GetRollsCount(s32 *AccessibleRollsCount, u8 *Input, umm InputSize, umm Stride)
{
    s32 ThreadIdx = blockIdx.x*blockDim.x + threadIdx.x; 
    
    s32 Lines = (s32)(InputSize/Stride);
    s32 LineSize = Stride - 1;
    s32 CharactersCount = Lines*LineSize;
    
    if(ThreadIdx < CharactersCount)
    {       
        s32 X = (ThreadIdx % LineSize);
        s32 Y = (ThreadIdx / LineSize);
        
        u8 *Char = Input + (Y*Stride + X);
        
        if(*Char == '@')
        {
            s32 MinX = Maximum(X - 1, 0);
            s32 MinY = Maximum(Y - 1, 0);
            s32 MaxX = Minimum(X + 1, (s32)LineSize);
            s32 MaxY = Minimum(Y + 1, (s32)Lines);
            
            s32 RollsCount = 0;
            for(s32 ScanY = MinY; ScanY <= MaxY; ScanY += 1)
            {
                for(s32 ScanX = MinX; ScanX <= MaxX; ScanX += 1)
                {
                    u8 *ScanChar = Input + (ScanY*Stride + ScanX);
                    if(!(ScanX == X && ScanY == Y))
                    {
                        RollsCount += !!(*ScanChar == '@');
                    }
                }
            }
            if(RollsCount < 4)
            {
                atomicAdd(AccessibleRollsCount, 1);
            }
        }
    }
    
}


ENTRY_POINT(EntryPoint)
{
    
    if(Params->ArgsCount >= 2)
    {
        str8 InputFile = OS_ReadEntireFileIntoMemory(Params->Args[1]);
        if(InputFile.Size)
        {
            umm LineSize = 0;
            
            for(umm Idx = 0; Idx < InputFile.Size; Idx += 1)
            {
                if(InputFile.Data[Idx] == '\n')
                {
                    LineSize = Idx;
                    break;
                }
            }
            umm Stride = LineSize + 1;
            Assert(InputFile.Size%Stride == 0);
            Assert(LineSize && InputFile.Data[LineSize] == '\n'); // NOTE(luca): We should check every line...
            
            CU_Check(cudaSetDevice(0));
            
            arena *CU_Arena = CU_ArenaAlloc(ThreadContext->Arena);
            s32 *AccessibleRollsCount = PushStruct(CU_Arena, s32);
            u8 *Input = PushArray(CU_Arena, u8, InputFile.Size);
            
            CU_Check(cudaMemcpy(Input, InputFile.Data, InputFile.Size, cudaMemcpyHostToDevice));
            
            umm Lines = InputFile.Size / Stride;
            u32 CharacterCount = (u32)(Lines*LineSize);
            
            u32 BlockSize = 256;
            u32 BlocksCount = (CharacterCount + BlockSize - 1) / BlockSize;
            
            GetRollsCount<<<BlocksCount, BlockSize>>>(AccessibleRollsCount, Input, InputFile.Size, Stride);
            CU_Check(cudaGetLastError()); 
            
            CU_Check(cudaDeviceSynchronize());
            
            s32 Count = 0;
            CU_Check(cudaMemcpy(&Count, AccessibleRollsCount, sizeof(Count), cudaMemcpyDeviceToHost));
            
            Log("The forklifts can access %d rolls.\n", Count); 
            
        }
        else
        {
            // TODO(luca): Loggign
        }
    }
    else
    {
        // TODO(luca): 
    }
    
    return 0;
}
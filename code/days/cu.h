/* date = December 12th 2025 2:58 pm */

#ifndef CU_H
#define CU_H

#define kernel __global__

#if AOC_INTERNAL
#else
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

#endif //CU_H

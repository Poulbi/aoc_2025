/* date = December 6th 2025 2:34 pm */

#ifndef OS_H
#define OS_H

#include "lanes.h"

global u8 LogBuffer[Kilobytes(64)];

#define ENTRY_POINT(Name) void *Name(void *Params)
typedef ENTRY_POINT(entry_point_func);
ENTRY_POINT(EntryPoint);

str8 OS_ReadEntireFileIntoMemory(char *FileName);
void OS_PrintFormat(char *Format, ...);
void OS_BarrierWait(barrier Barrier);
void OS_SetThreadName(str8 ThreadName);
void* OS_Allocate(umm Size);

#if OS_LINUX
# include "linux.c"
#elif OS_WINDOWS
# include "windows.c"
#endif

#endif //OS_H

/* date = December 6th 2025 3:00 am */

#ifndef LANES_H
#define LANES_H
typedef u64 barrier;

typedef struct lane_context lane_context;
struct lane_context
{
    s64 LaneCount;
    s64 LaneIndex;
    
    u64 *SharedStorage;
    barrier Barrier;
};

#endif //LANES_H

#ifndef PROJECT_NCNN_SDRAM_PROFILE_H_
#define PROJECT_NCNN_SDRAM_PROFILE_H_

#include <stddef.h>

struct ncnn_sdram_stats {
	size_t capacity;
	size_t page_size;
	size_t baseline;
	size_t peak;
	size_t final_used;
	size_t failed_allocations;
	size_t other_heap_allocations;
};

int ncnn_sdram_profile_begin(void);
void ncnn_sdram_profile_end(struct ncnn_sdram_stats *result);
int ncnn_sdram_profile_selftest(void);

#endif

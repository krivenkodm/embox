/* Project-local page-allocation hooks; enabled only by this board's linker
 * flags. Sample after EVERY allocation, so short-lived workspaces count too.
 * The peak includes heap segment metadata, alignment and unused segment space;
 * it excludes the page allocator's fixed control area (capacity reports that).
 */
#include <stddef.h>
#include <string.h>
#include <kernel/sched/sched_lock.h>
#include <mem/page.h>

#include "sdram_profile.h"

extern struct page_allocator *__heap_fixed_pgallocator;
extern void *__real_page_alloc(struct page_allocator *, size_t);
extern void *__real_page_alloc_zero(struct page_allocator *, size_t);

static int active;
static struct ncnn_sdram_stats stats;

static size_t used(void) {
	return stats.capacity - __heap_fixed_pgallocator->free;
}

static void sample(struct page_allocator *allocator, void *result) {
	if (!active) {
		return;
	}
	if (allocator == __heap_fixed_pgallocator) {
		const size_t current = used();
		if (current > stats.peak) {
			stats.peak = current;
		}
		if (!result) {
			++stats.failed_allocations;
		}
	} else if (result) {
		++stats.other_heap_allocations;
	}
}

void *__wrap_page_alloc(struct page_allocator *allocator, size_t count) {
	void *result;
	sched_lock();
	result = __real_page_alloc(allocator, count);
	sample(allocator, result);
	sched_unlock();
	return result;
}

void *__wrap_page_alloc_zero(struct page_allocator *allocator, size_t count) {
	void *result;
	sched_lock();
	result = __real_page_alloc_zero(allocator, count);
	sample(allocator, result);
	sched_unlock();
	return result;
}

int ncnn_sdram_profile_begin(void) {
	struct page_allocator *allocator = __heap_fixed_pgallocator;
	sched_lock();
	if (active || !allocator) {
		sched_unlock();
		return -1;
	}
	memset(&stats, 0, sizeof(stats));
	stats.capacity = allocator->pages_n * allocator->page_size;
	stats.page_size = allocator->page_size;
	stats.baseline = stats.peak = used();
	active = 1;
	sched_unlock();
	return 0;
}

void ncnn_sdram_profile_end(struct ncnn_sdram_stats *result) {
	sched_lock();
	stats.final_used = used();
	active = 0;
	*result = stats;
	sched_unlock();
}

/* Check both linker hooks with a known three-page peak and full release.
 * A missing --wrap must not silently report a plausible zero-byte peak. */
int ncnn_sdram_profile_selftest(void) {
	void *first, *second;
	int first_observed;
	struct ncnn_sdram_stats result;
	if (ncnn_sdram_profile_begin() != 0) {
		return -1;
	}
	first = page_alloc(__heap_fixed_pgallocator, 1);
	first_observed = stats.peak == stats.baseline + stats.page_size;
	second = page_alloc_zero(__heap_fixed_pgallocator, 2);
	if (second) {
		page_free(__heap_fixed_pgallocator, second, 2);
	}
	if (first) {
		page_free(__heap_fixed_pgallocator, first, 1);
	}
	ncnn_sdram_profile_end(&result);
	return first && second && first_observed && result.peak == result.baseline + 3 * result.page_size
			&& result.final_used == result.baseline ? 0 : -1;
}

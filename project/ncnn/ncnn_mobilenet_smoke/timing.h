#ifndef PROJECT_NCNN_MOBILENET_TIMING_H_
#define PROJECT_NCNN_MOBILENET_TIMING_H_

#include <stdint.h>
#include <time.h>

class Timer {
public:
	Timer() { reset(); }
	void reset() { valid_ = clock_gettime(CLOCK_MONOTONIC, &start_) == 0; }
	long milliseconds() const {
		struct timespec end;
		if (!valid_ || clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
			return -1;
		}
		return static_cast<long>(((int64_t)(end.tv_sec - start_.tv_sec) * 1000000000
				+ end.tv_nsec - start_.tv_nsec) / 1000000);
	}
private:
	struct timespec start_;
	bool valid_;
};

struct RunTimings {
	long param = -1, model = -1, input = -1, extract = -1, verify = -1, cleanup = -1;
};

/* Declare before the Net so this destructor runs after every NCNN object. */
class CleanupTimer {
public:
	explicit CleanupTimer(long &result) : result_(result), armed_(false) {}
	void start() { timer_.reset(); armed_ = true; }
	~CleanupTimer() { if (armed_) result_ = timer_.milliseconds(); }
private:
	Timer timer_;
	long &result_;
	bool armed_;
};

#endif

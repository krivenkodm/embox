#ifndef PROJECT_NCNN_BOUNDED_READER_H_
#define PROJECT_NCNN_BOUNDED_READER_H_

#include <stddef.h>
#include <string.h>
#include <datareader.h>

/* NCNN's default memory reader has no size argument. Keep reads/references
 * inside one validated QSPI blob and latch any bounds failure. */
class BoundedReader : public ncnn::DataReader {
public:
	BoundedReader(const unsigned char *data, size_t size)
		: data_(data), size_(size), position_(0), references_(0), failed_(false) {}

	size_t read(void *buffer, size_t size) const override {
		if (!available(size)) {
			return 0;
		}
		memcpy(buffer, data_ + position_, size);
		position_ += size;
		return size;
	}

	size_t reference(size_t size, const void **buffer) const override {
		if (!available(size)) {
			return 0;
		}
		*buffer = data_ + position_;
		position_ += size;
		references_ += size;
		return size;
	}

	bool complete() const { return !failed_ && position_ == size_; }
	size_t referenced() const { return references_; }

private:
	bool available(size_t size) const {
		if (failed_ || size > size_ - position_) {
			failed_ = true;
			return false;
		}
		return true;
	}
	const unsigned char *data_;
	const size_t size_;
	mutable size_t position_;
	mutable size_t references_;
	mutable bool failed_;
};

#endif

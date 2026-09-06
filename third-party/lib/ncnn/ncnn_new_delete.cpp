// SPDX-License-Identifier: BSD-2-Clause

#include <stddef.h>

void *operator new(size_t, void *ptr) {
	return ptr;
}

void operator delete(void *, void *) {
}

void *operator new[](size_t, void *ptr) {
	return ptr;
}

void operator delete[](void *, void *) {
}

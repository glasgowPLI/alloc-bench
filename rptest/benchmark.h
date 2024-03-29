
#include <string.h>
#include <stdlib.h>
#ifndef __APPLE__
#  include <malloc.h>
#endif

#if defined(BUMPALLOC)
#  include "bump_alloc.h"
#endif

int
benchmark_initialize() {
	return 0;
}

int
benchmark_finalize(void) {
	return 0;
}

int
benchmark_thread_initialize(void) {
	return 0;
}

int
benchmark_thread_finalize(void) {
	return 0;
}

#if defined(BUMPALLOC)
void*
benchmark_malloc(size_t alignment, size_t size) {
  return bump_alloc(size);
} 

extern void
benchmark_free(void* ptr) {
	bump_free(ptr);
}

#else   /* standard malloc */
void*
benchmark_malloc(size_t alignment, size_t size) {
	// memset/calloc to ensure all memory is touched!
	if (alignment != 0) {
		#if defined(__MACH__)
		void* ptr = NULL;
		posix_memalign(&ptr, alignment, size);
		#else
		void* ptr = memalign(alignment, size);
		#endif
		if (ptr != NULL) memset(ptr,0xCD,size);
		return ptr;
	}
	else {
		return calloc(1,size);
	}
}

extern void
benchmark_free(void* ptr) {
	free(ptr);
}
#endif

const char*
benchmark_name(void) {
	return "crt";
}

void
benchmark_thread_collect(void) {
}

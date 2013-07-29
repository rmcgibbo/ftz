/*
* Flush arrays of denormalized numbers to zero.
*/

#include <mmintrin.h>    /* SSE */
#include <pmmintrin.h>   /* DENORMAL_MODE */
#include <float.h>
#include <stdio.h>

#include "ftz.h"

/*
 * Flush denormalized single-precision floating point numbers to zero, without
 * SSE.
 */ 
static inline float* fftz_noalign(float* x, size_t length) {
  int i;
  float* X = x;
  for (i = 0; i < length; i++, X++) {
    *X *= (*X > FLT_MIN) || (*X < -FLT_MIN);
  }
  return X;
}

/*
 * Flush denormalized double-precision floating point numbers to zero, without
 * SSE
 */ 
static inline double* dftz_noalign(double* x, size_t length) {
  int i;
  double* X = x;
  for (i = 0; i < length; i++, X++) {
    *X *= (*X > DBL_MIN) || (*X < -DBL_MIN);
  }
  return X;
}

/*
 * Flush denormalized single-precision floating point numbers to zero.
 */ 
int fftz(float* x, size_t length) {
  size_t i;
  float* y;
  float* X = x;
  size_t offset = (size_t) x % 16;
  size_t increment;
  size_t remaining = length;
  __m128 f;

  // Flush the floats before a 16 byte boundary by hand
  if (offset != 0) {
    increment = (16 - offset) / 4;
    if (increment > length)
      increment = length;
    X = fftz_noalign(X, increment);
    remaining -= increment;
  }

  if (length == 0) {
    return 1;
  }

  int flushMode = _MM_GET_FLUSH_ZERO_MODE();
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  // Use SSE to process floats 4 at a time.
  // Note: this code MUST be compiled at -O0 (without)
  // optimizations, otherwise the add opteration
  // will get removed by the compiler
  __m128 one = _mm_set1_ps(0.0f);
  for (i = 0; i < remaining / 4; i++) {
    f = _mm_load_ps(X);  // load 4 floats from mem.
    f = _mm_add_ps(f, one);     // add zero. this triggers flush.
    _mm_store_ps(X, f);         // store back to mem.
    X = X + 4;
  }
  _MM_SET_FLUSH_ZERO_MODE(flushMode);

  // Do the remaining elements without SSE, if the
  // length is not a multiple of four
  increment = length - (X - x);
  fftz_noalign(X, increment);

  return 1;
}

/*
 * Flush denormalized double-precision floating point numbers to zero.
 */ 
int dftz(double* x, size_t length) {
  size_t i;
  double* X = x;
  size_t offset = (size_t) x % 16;
  size_t increment;
  size_t remaining = length;
  __m128d f;

  if (offset != 0) {
    increment = (16 - offset) / 8;
    if (increment > length)
      increment = length;
    X = dftz_noalign(X, increment);
    remaining -= increment;
  }

  if (length == 0) {
    return 1;
  }

  int flushMode = _MM_GET_FLUSH_ZERO_MODE();
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  __m128d one = _mm_set1_pd(0.0);
  for (i = 0; i < remaining / 2; i++) {
    f = _mm_load_pd(X);         // load 2 floats from mem.
    f = _mm_add_pd(f, one);     // add zero. this triggers flush.
    _mm_store_pd(X, f);         // store back to mem.
    X = X + 2;
  }
  _MM_SET_FLUSH_ZERO_MODE(flushMode);

  // Do the remaining elements without SSE, if the
  // length is not a multiple of four
  increment = length - (X - x);
  dftz_noalign(X, increment);

  return 1;

}
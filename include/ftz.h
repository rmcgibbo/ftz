#ifndef _FTZ_H
#define _FTZ_H

/*
 * Flush denormalized single-precision floating point numbers to zero.
 */ 
int fftz(float* x, size_t length);


/*
 * Flush denormalized double-precision floating point numbers to zero.
 */ 
int dftz(double* x, size_t length);

#endif
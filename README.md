Flush Denorms to Zero
---------------------

Denormalized numbers ("denorms") are extremely small numbers (less than
1.2-38 for single precision or 2.2e-308 in double precision) which are
handled poorly on most modern archicectures. Arithmetic involving denorms
can be up to 100x slower than arithmetic on standard floating point
numbers. Denorms can also give annoying behavior, like overflowing when
you take their reciprocal.

This package implements a single function to flush denorms to zero in float32
and float64 numpy arrays using SSE instructions.
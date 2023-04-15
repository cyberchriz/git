## Heap Allocation Logger
dependencies: `<iostream>`, `<boost/stacktrace.hpp>`;

usage:
1. first set a preprocessor flag `#define MEMLOG`
2. then `#include <memlog.h>`, alternatively include as part of the `<utilities.h>` library
This will overide the `new` keyword: anytime memory is allocated on the heap this will be logged to the console
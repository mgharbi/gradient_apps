#include <TH/TH.h>
#include <stdio.h>

extern "C" {

int dummy_forward(
    THFloatTensor *data)
  THError("CPU not implemented");
  return 1;
}

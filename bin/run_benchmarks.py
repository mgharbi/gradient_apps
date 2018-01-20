#!/usr/bin/env python
import gapps.benchmarks as b

def main():
  res = [
    # b.BackwardConv2d(False),
    # b.BackwardConv2d(False, True),
    # b.BackwardConv2d(True),
    # b.BackwardConv2d(True, True),
    # b.BilateralLayer(True),
    # b.BilateralLayer(True, True),
    # b.BilateralLayer(False),
    # b.BilateralLayer(False, True),
    b.BilateralSliceApply(True),
    b.BilateralSliceApply(True, True),
    # b.SpatialTransformer(False),
    # b.SpatialTransformer(False, True),
    # b.SpatialTransformer(True),
    # b.SpatialTransformer(True, True),
    # b.VGG(False),
    # b.VGG(False, True),
    # b.VGG(True),
    # b.VGG(True, True),
  ]

  for r in res:
    print r()


if __name__ == "__main__":
  main()

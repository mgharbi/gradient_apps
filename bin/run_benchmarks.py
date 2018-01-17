#!/usr/bin/env python
import gapps.benchmarks as b

def main():
  res = [
    b.BilateralLayer(True),
    b.BilateralLayer(True, True),
    # b.BilateralLayer(False),
    # b.BilateralLayer(False, True),
    # b.SpatialTransformer(False),
    # b.SpatialTransformer(False, True),
    # b.SpatialTransformer(True),
    # b.SpatialTransformer(True, True),
    # b.VGG(False, True),
    # b.VGG(False),
  ]

  for r in res:
    print r()


if __name__ == "__main__":
  main()

#!/usr/bin/env python
import gapps.benchmarks as b

def main():
  res = [
    # b.SpatialTransformer(False),
    # b.SpatialTransformer(False, True),
    b.SpatialTransformer(True),
    b.SpatialTransformer(True, True),
    # b.VGG(True),
    # b.VGG(True, True),
  ]

  for r in res:
    print r()


if __name__ == "__main__":
  main()

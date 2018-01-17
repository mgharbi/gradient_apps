#!/usr/bin/env python
import gapps.benchmarks as b

def main():
  res = [
    b.SpatialTransformer(False),
    b.SpatialTransformer(True),
    b.SpatialTransformer(False, True),
    b.SpatialTransformer(True, True),
  ]

  for r in res:
    print r()


if __name__ == "__main__":
  main()

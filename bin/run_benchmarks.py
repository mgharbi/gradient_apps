#!/usr/bin/env python
import gapps.benchmarks as b

def main():
  res = [
    # b.BackwardConv2d(False),
    # b.BackwardConv2d(False, True),
    b.BackwardConv2d(True),
    b.BackwardConv2d(True, True),
    # b.BilateralLayer(True),
    # b.BilateralLayer(True, True),
    # b.BilateralLayer(False),
    # b.BilateralLayer(False, True),
    # b.BilateralSliceApply(True, "halide"),
    # b.BilateralSliceApply(True, "manual"),
    # b.BilateralSliceApply(True, "pytorch"),
    # b.BilateralSliceApply(False, "halide"),
    # b.BilateralSliceApply(False, "pytorch"),
    # b.SpatialTransformer(False),
    # b.SpatialTransformer(False, True),
    # b.SpatialTransformer(True),
    # b.SpatialTransformer(True, True),
    # b.Flownet(True, "nvidia"),
    # b.Flownet(True, "halide"),
    # b.Flownet(True, "pytorch"),
    # b.Flownet(False, "pytorch"),
    # b.Flownet(False, "halide"),
    # b.Flownet(False),
    # b.Flownet(False, True),
    # b.VGG(False),
    # b.VGG(False, True),
    # b.VGG(True),
    # b.VGG(True, True),
  ]

  for r in res:
    print(r())


if __name__ == "__main__":
  main()

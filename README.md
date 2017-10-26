# Applications for gradient Halide

### Dependencies

- pytorch
- py.test
- halide

### Building

Set the `HALIDE_DIR` env var to your Halide root directory. Then run:

    cd gapps
    make

### Adding a new operator

See 'dummy' for an example. This is done in 5 steps:

1. Write a Halide generator e.g. `src/dummy.hl.cxx`.

2. Declare a CFFI entry point in `src/operators.h`.

3. Write the PyTorch C interface in `src/operator.cxx`.

4. Register your new operator with Autograd in functions/operators.py.

5. Add an entry in the Makefile (under OPS)


### Running tests

Run with stdout prints:

    cd gapps py.test --capture=no test

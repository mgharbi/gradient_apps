.PHONY: gapps clean gradient-halide

gapps: gradient-halide
	@$(MAKE) -C $@

gradient-halide:
	@$(MAKE) -C $@

clean:
	$(MAKE) -C gapps clean

# TODO: make server

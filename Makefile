.PHONY: gapps clean
gapps:
	@$(MAKE) -C $@

clean:
	$(MAKE) -C gapps clean

# TODO: make server

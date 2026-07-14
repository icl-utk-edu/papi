CYPAPI_REPO = https://github.com/icl-utk-edu/cyPAPI.git
CYPAPI_DIR  = cyPAPI

.PHONY: all install clean

all: install

install:
	@if [ -d "$(CYPAPI_DIR)" ]; then \
		echo "$(CYPAPI_DIR) already exists, skipping clone."; \
	else \
		git clone $(CYPAPI_REPO) $(CYPAPI_DIR); \
	fi
	$(MAKE) -C $(CYPAPI_DIR) install

clean:
	rm -rf $(CYPAPI_DIR)

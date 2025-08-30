BEAKER_USER := $(shell beaker account whoami --format=json | jq -r '.[0].name')
TIMESTAMP := $(shell date "+%Y%m%d%H%M%S")
WORKSPACE = ai2/google_benchmarks

.PHONY : beaker-image
beaker-image :
	docker build -f Dockerfile -t maxtext .
	beaker image create maxtext --name maxtext-tmp --workspace $(WORKSPACE)
	beaker image rename $(BEAKER_USER)/maxtext maxtext-$(TIMESTAMP) >/dev/null 2>&1 || true
	beaker image rename $(BEAKER_USER)/maxtext-tmp maxtext

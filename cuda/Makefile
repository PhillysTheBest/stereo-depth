NVCC ?= nvcc
# T4 (sm_75) for Colab, plus sm_86 and sm_89 for common consumer GPUs.
ARCHS := -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCCFLAGS := -O3 -std=c++17 -Xcompiler -Wall $(ARCHS)

# Compiles the kernels without PyTorch. CI runs this target; it needs nvcc but no GPU.
build/selftest: sgm/sgm.cu sgm/sgm.h tests/selftest.cu
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ sgm/sgm.cu tests/selftest.cu

# Needs a GPU.
.PHONY: test
test: build/selftest
	./build/selftest

.PHONY: clean
clean:
	rm -rf build dist *.egg-info

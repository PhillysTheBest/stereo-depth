NVCC ?= nvcc
# T4 (sm_75) for Colab, plus sm_86 and sm_89 for common consumer GPUs.
ARCHS := -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89

build/hello: sgm/hello.cu
	mkdir -p build
	$(NVCC) -O2 $(ARCHS) -o $@ $<

.PHONY: clean
clean:
	rm -rf build

# Build inside the shadps4-dev distrobox on the BC-250 (Vulkan headers + glslang live there):
#
#   distrobox enter shadps4-dev -- make
#   ./bc250-present-probe                 # all correctness cases + 1080p timing
#   ./bc250-present-probe rt 1920 1080    # one custom case
#
# No libdrm, no Kyty objects: links only against the Vulkan loader.

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
GLSLANG  ?= glslangValidator

SHADERS := detile_rt copy_linear
SPV_HDRS := $(addsuffix .spv.h,$(SHADERS))

all: bc250-present-probe

# One combined header holding every shader's SPIR-V as a named uint32_t array.
shaders.h: $(SPV_HDRS)
	cat $(SPV_HDRS) > $@

%.spv.h: %.comp
	$(GLSLANG) -V --vn $(subst -,_,$*)_spv $< -o $@

bc250-present-probe: present_probe.cpp shaders.h
	$(CXX) $(CXXFLAGS) present_probe.cpp -o $@ -lvulkan

clean:
	rm -f bc250-present-probe shaders.h $(SPV_HDRS)

.PHONY: all clean

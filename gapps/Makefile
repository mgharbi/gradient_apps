TORCH_INC ?= `python -c 'import torch.utils.ffi as ffi; print("-I"+" -I".join(ffi._setup_wrapper(True)[1]))'`
HALIDE_DIR ?= 
HL_TARGET ?= host

CXX ?= g++
CXXFLAGS += -O3 -g -std=c++11 -fno-rtti -I$(HALIDE_DIR)/include/
LDFLAGS ?=
LDFLAGS += $(HALIDE_DIR)/lib/libHalide.a -lpthread -ldl -lcurses -lz
ifeq ($(UNAME), Darwin)
CXXFLAGS += -fvisibility=hidden
endif

ifeq ($(UNAME), Darwin)
DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH):$(HALIDE_DIR)/bin
else
endif

# Cuda config
NVCC = nvcc -std c++11 -g #-G  -pg
NVFLAGS = -x cu -Xcompiler -fPIC -I$(SRC_DIR) \
					-gencode=arch=compute_30,code=\"sm_30,compute_30\" -expt-relaxed-constexpr -Wno-deprecated-gpu-targets -ftz=true --ptxas-options=-v -lineinfo 

SRC_DIR = src
BUILD_DIR = build
EXT_DIR = _ext
OPS = bilateral_layer_forward #bilateral_layer_backward
OPS_LIBS = $(addsuffix .a, $(addprefix $(BUILD_DIR)/, $(OPS)))

all: cffi

cffi: $(OPS_LIBS)
	python build.py

# build/kernels.so: src/sample_weighting_kernel.cu src/sample_weighting_kernel.h build
# 	$(NVCC) -c  $< -o $@ $(NVFLAGS) $(TORCH_INC) 
#
# $(OPS_LIB): $(OPS_LIBS)
# 	ar rvs $@ $^

$(BUILD_DIR)/%.a: $(BUILD_DIR)/%
	DYLD_LIBRARY_PATH=$(HALIDE_DIR)/bin LD_LIBRARY_PATH=$(HALIDE_DIR)/bin ./$< -g $(basename $(@F)) -e static_library,h,html -o $(BUILD_DIR) target=$(HL_TARGET)

# Generators
$(BUILD_DIR)/%: $(SRC_DIR)/%.hl.cxx build
	echo $(HALIDE_DIR)
	$(CXX) $< $(HALIDE_DIR)/tools/GenGen.cpp  $(CXXFLAGS) $(LDFLAGS) -o $@

build:
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(EXT_DIR)

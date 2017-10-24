TORCH_INC ?= `python -c 'import torch.utils.ffi as ffi; print("-I"+" -I".join(ffi._setup_wrapper(True)[1]))'`
HALIDE_DIR ?= $(HOME)/projects/gradient_halide
HALIDE_BIN_PATH ?= $(HALIDE_DIR)/lib
HALIDE_SRC_PATH ?= $(HALIDE_DIR)
HL_TARGET ?= host

LDFLAGS ?=
LDFLAGS += -ldl -lpthread -lz -lHalide

CXX ?= g++
CXXFLAGS += -std=c++11 -I $(HALIDE_BIN_PATH)/include/ -I $(HALIDE_SRC_PATH)/tools/
ifeq ($(UNAME), Darwin)
CXXFLAGS += -fvisibility=hidden
endif

LIB_HALIDE = $(HALIDE_BIN_PATH)/lib/libHalide.a
GENERATOR_DEPS ?= $(HALIDE_BIN_PATH)/lib/libHalide.a $(HALIDE_BIN_PATH)/include/Halide.h $(HALIDE_SRC_PATH)/tools/GenGen.cpp

# Cuda config
NVCC = nvcc -std c++11 -g #-G  -pg
NVFLAGS = -x cu -Xcompiler -fPIC -I$(SRC_DIR) \
					-gencode=arch=compute_30,code=\"sm_30,compute_30\" -expt-relaxed-constexpr -Wno-deprecated-gpu-targets -ftz=true --ptxas-options=-v -lineinfo 

SRC_DIR = src
BUILD_DIR = build
EXT_DIR = _ext
# SOURCES = 
OPS_LIB = $(BUILD_DIR)/operators.so
OPS = dummy
OPS_OBJ = $(addsuffix $(addprefix $OPS, $(BUILD_DIR)/), .o)

all: cffi

cffi: build/kernels.so
	python build.py

# build/kernels.so: src/sample_weighting_kernel.cu src/sample_weighting_kernel.h build
# 	$(NVCC) -c  $< -o $@ $(NVFLAGS) $(TORCH_INC) 
#
$(OPS_LIB): $(OPS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ -shared $(LDFLAGS)

$(BUILD_DIR)/%.o: $(BUILD_DIR)/%
	./$<

# Generators
$(BUILD_DIR)/%: $(SRC_DIR)/%.hl.cxx
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# / g++ lesson_10*generate.cpp -g -std=c++11 -I ../include -L ../bin -lHalide -lpthread -ldl -o lesson_10_generate
# // LD_LIBRARY_PATH=../bin ./lesson_10_generate
# // g++ lesson_10*run.cpp lesson_10_halide.a -lpthread -ldl -o lesson_10_run
# // ./lesson_10_run
# // g++ lesson_15*.cpp ../tools/GenGen.cpp -g -std=c++11 -fno-rtti -I ../include -L ../bin -lHalide -lpthread -ldl -o lesson_15_generate
# // bash lesson_15_generators_usage.sh

build:
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(EXT_DIR)

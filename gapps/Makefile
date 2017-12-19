TORCH_INC ?= `python -c 'import torch.utils.ffi as ffi; print("-I"+" -I".join(ffi._setup_wrapper(True)[1]))'`
HALIDE_DIR ?=
AUTO_SCHEDULE = false
BUILD_TYPE ?= release

CXX ?= g++
CXXFLAGS += -std=c++11 -fno-rtti
INCLUDE=-I$(HALIDE_DIR)/include/
LDFLAGS ?=
LDFLAGS += $(HALIDE_DIR)/lib/libHalide.a -lpthread -ldl -lcurses -lz

# Cuda config
NVCC = nvcc -std c++11 #-G  -pg
NVFLAGS = -x cu -Xcompiler -fPIC -I$(SRC_DIR) \
					-gencode=arch=compute_30,code=\"sm_30,compute_30\" \
					-expt-relaxed-constexpr -Wno-deprecated-gpu-targets \
					-ftz=true --ptxas-options=-v -lineinfo
CUDA_LDFLAGS= -L/usr/local/cuda/lib64 -lcuda -lcudart

TARGET_FEATURES ?=

ifeq ($(BUILD_TYPE), profile)
TARGET_FEATURES = -profile
endif

ifeq ($(BUILD_TYPE), debug)
CXXFLAGS += -g -rdynamic
NVFLAGS += -g -G -lineinfo
else
CXXFLAGS += -O3
endif

ifeq ($(UNAME), Darwin)
CXXFLAGS += -fvisibility=hidden
endif

ifeq ($(UNAME), Darwin)
DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH):$(HALIDE_DIR)/bin
else
endif


HALIDE_GEN = $(HALIDE_DIR)/tools/GenGen.cpp

SRC_DIR = src
BUILD_DIR = build
EXT_DIR = _ext

# The makefile assumes that each op that has a backward also has a forward
OPS = \
			deconv_cg_init_forward deconv_cg_init_backward deconv_cg_iter_forward deconv_cg_iter_backward \
			deconv_cg_weight_forward deconv_cg_weight_backward \
			# learnable_demosaick_forward learnable_demosaick_backward \
			# naive_demosaick_forward naive_demosaick_backward \
			# conv1d_forward conv1d_backward conv1d_manual_backward \
			# histogram_forward histogram_backward \
			# soft_histogram_forward soft_histogram_backward

OPS_LIBS = $(addsuffix .a, $(addprefix $(BUILD_DIR)/, $(OPS)))
CUDA_OPS_LIBS = $(addsuffix _cuda.a, $(addprefix $(BUILD_DIR)/, $(OPS)))

_ext/operators/_operators.so: $(OPS_LIBS) #$(CUDA_OPS_LIBS)
	@echo Python CFFI wrapper
	@python build.py

$(BUILD_DIR)/%.a: $(BUILD_DIR)/%
	@echo Op $(subst $(BUILD_DIR)/,, $@)
	@cd $(BUILD_DIR); \
	DYLD_LIBRARY_PATH=$(HALIDE_DIR)/bin LD_LIBRARY_PATH=$(HALIDE_DIR)/bin \
										./$(basename $(@F)) -g $(basename $(@F)) \
										-e static_library,h,pytorch_wrapper,html\
										-o . target=host$(TARGET_FEATURES) \
										auto_schedule=$(AUTO_SCHEDULE)

$(BUILD_DIR)/%_cuda.a: $(BUILD_DIR)/% 
	@$(eval name = $(subst _cuda,,$(basename $(@F))))
	@echo "Op (CUDA) $(name)"
	@cd $(BUILD_DIR); \
	DYLD_LIBRARY_PATH=$(HALIDE_DIR)/bin LD_LIBRARY_PATH=$(HALIDE_DIR)/bin \
										./$(name) -g $(name) -f $(name)_cuda \
										-e static_library,h,pytorch_wrapper,html \
										-o . target=host-cuda-cuda_capability_35$(TARGET_FEATURES) \
										auto_schedule=$(AUTO_SCHEDULE)

# Generators, prevent auto-deletion by make
GENERATORS = $(addprefix $(BUILD_DIR)/, $(OPS))
.SECONDARY: $(GENERATORS)

$(BUILD_DIR)/%: $(SRC_DIR)/%.hl.cxx $(HALIDE_GEN) $(BUILD_DIR)
	@echo Generator $(subst $(BUILD_DIR)/,, $@)
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(HALIDE_GEN) $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -MMD -MP -o $(basename $@)

$(BUILD_DIR):
	@echo Making $@ dir
	@mkdir -p $@

# C++ target to test the halide code independently of pytorch
$(BUILD_DIR)/tester: $(SRC_DIR)/_tester.cpp $(BUILD_DIR)/learnable_demosaick_backward_cuda.a
	# $(BUILD_DIR)/learnable_demosaick_forward_cuda.a $(BUILD_DIR)/learnable_demosaick_forward.a \
	# $(BUILD_DIR)/learnable_demosaick_backward_cuda.a $(BUILD_DIR)/learnable_demosaick_backward.a
	$(CXX) $(CXXFLAGS) $(INCLUDE) -I/usr/local/cuda/include -I$(BUILD_DIR) $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

$(BUILD_DIR)/tester2: $(SRC_DIR)/_tester2.cpp $(BUILD_DIR)/conv1d_manual_backward.a
	$(CXX) $(CXXFLAGS) $(INCLUDE) -I/usr/local/cuda/include -I$(BUILD_DIR) $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

clean:
	$(RM) -r $(BUILD_DIR) $(EXT_DIR)

DEPS = $(wildcard $(BUILD_DIR)/*.d)

-include $(DEPS)

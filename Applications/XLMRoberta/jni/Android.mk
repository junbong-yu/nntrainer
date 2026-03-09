LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

ML_API_COMMON_INCLUDES := ${NNTRAINER_ROOT}/ml_api_common/include
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/builddir/android_build_result/include/nntrainer 
NNTRAINER_SRC_INCLUDES := $(NNTRAINER_ROOT)/nntrainer
NNTRAINER_TENSOR_INCLUDES := $(NNTRAINER_ROOT)/nntrainer/tensor
NNTRAINER_LAYERS_INCLUDES := $(NNTRAINER_ROOT)/nntrainer/layers
NNTRAINER_UTILS_INCLUDES := $(NNTRAINER_ROOT)/nntrainer/utils

CAUSALLM_PATH := $(NNTRAINER_ROOT)/Applications/CausalLM

# Prebuilt nntrainer libraries
include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

# Tokenizer library
include $(CLEAR_VARS)
LOCAL_MODULE := tokenizers_c
LOCAL_SRC_FILES := ../lib/libtokenizers_android_c.a
include $(PREBUILT_STATIC_LIBRARY)

# Build XLMRoberta executable
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_LDFLAGS += -fexceptions -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntr_xlmroberta
LOCAL_LDLIBS := -llog -landroid -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1

# XLMRoberta source files
LOCAL_SRC_FILES := ../main.cpp \
    ../xlm_roberta.cpp \
    ../position.cpp \
    ../token_type.cpp \
    ../util.cpp \
    ../fc_layer_single.cpp \
    $(CAUSALLM_PATH)/models/causal_lm.cpp \
    $(CAUSALLM_PATH)/models/transformer.cpp \
    $(CAUSALLM_PATH)/models/embedding.cpp \
    $(CAUSALLM_PATH)/llm_util.cpp \
    $(CAUSALLM_PATH)/layers/embedding_layer.cpp \
    $(CAUSALLM_PATH)/layers/embedding_pooling_layer.cpp \
    $(CAUSALLM_PATH)/layers/embedding_normalize_layer.cpp \
    $(CAUSALLM_PATH)/layers/mha_core.cpp \
    $(CAUSALLM_PATH)/layers/reshaped_rms_norm.cpp \
    $(CAUSALLM_PATH)/layers/rms_norm.cpp \
    $(CAUSALLM_PATH)/layers/swiglu.cpp \
    $(CAUSALLM_PATH)/layers/tie_word_embedding.cpp \
    $(CAUSALLM_PATH)/layers/lm_head.cpp \
    $(CAUSALLM_PATH)/layers/qkv_layer.cpp \
    $(CAUSALLM_PATH)/huggingface_tokenizer.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) \
    $(LOCAL_PATH)/.. \
    $(CAUSALLM_PATH) \
    $(CAUSALLM_PATH)/layers \
    $(CAUSALLM_PATH)/models \
    $(NNTRAINER_SRC_INCLUDES) \
    $(NNTRAINER_TENSOR_INCLUDES) \
    $(NNTRAINER_LAYERS_INCLUDES) \
    $(NNTRAINER_UTILS_INCLUDES)

include $(BUILD_EXECUTABLE)


# Build nntr_quantize executable
include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_CFLAGS += -std=c++17 -Ofast -mcpu=cortex-a53 -Ilz4-nougat/lib -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_LDFLAGS += -Llz4-nougat/lib/obj/local/$(TARGET_ARCH_ABI)/
LOCAL_CXXFLAGS += -std=c++17 -frtti
LOCAL_CFLAGS += -pthread -fexceptions -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_LDFLAGS += -fexceptions -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1 -mtune=cortex-a76 -O3 -ffast-math
LOCAL_MODULE_TAGS := optional
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntr_quantize
LOCAL_LDLIBS := -llog -landroid -fopenmp -static-openmp -DENABLE_FP16=1 -DUSE__FP16=1 -D__ARM_NEON__=1 -march=armv8.2-a+fp16+dotprod+i8mm -DUSE_NEON=1

# Source files
LOCAL_SRC_FILES := ../quantize.cpp \
    ../xlm_roberta.cpp \
    ../position.cpp \
    ../token_type.cpp \
    ../util.cpp \
    ../fc_layer_single.cpp \
    $(CAUSALLM_PATH)/models/causal_lm.cpp \
    $(CAUSALLM_PATH)/models/transformer.cpp \
    $(CAUSALLM_PATH)/models/embedding.cpp \
    $(CAUSALLM_PATH)/llm_util.cpp \
    $(CAUSALLM_PATH)/layers/embedding_layer.cpp \
    $(CAUSALLM_PATH)/layers/embedding_pooling_layer.cpp \
    $(CAUSALLM_PATH)/layers/embedding_normalize_layer.cpp \
    $(CAUSALLM_PATH)/layers/mha_core.cpp \
    $(CAUSALLM_PATH)/layers/reshaped_rms_norm.cpp \
    $(CAUSALLM_PATH)/layers/rms_norm.cpp \
    $(CAUSALLM_PATH)/layers/swiglu.cpp \
    $(CAUSALLM_PATH)/layers/tie_word_embedding.cpp \
    $(CAUSALLM_PATH)/layers/lm_head.cpp \
    $(CAUSALLM_PATH)/layers/qkv_layer.cpp \
    $(CAUSALLM_PATH)/huggingface_tokenizer.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c

LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES) \
    $(LOCAL_PATH)/.. \
    $(CAUSALLM_PATH) \
    $(CAUSALLM_PATH)/layers \
    $(CAUSALLM_PATH)/models \
    $(NNTRAINER_SRC_INCLUDES) \
    $(NNTRAINER_TENSOR_INCLUDES) \
    $(NNTRAINER_LAYERS_INCLUDES) \
    $(NNTRAINER_UTILS_INCLUDES)

include $(BUILD_EXECUTABLE)

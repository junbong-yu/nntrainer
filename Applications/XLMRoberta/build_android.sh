#!/bin/bash

# Build script for XLMRoberta Android application
set -e

# Check if NDK path is set
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set. Please set it to your Android NDK path."
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi

# Set NNTRAINER_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export NNTRAINER_ROOT

echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"

sudo rm -rf "$NNTRAINER_ROOT/builddir"
# Step 1: Build nntrainer for Android if not already built
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    echo "Building nntrainer for Android..."
    cd "$NNTRAINER_ROOT"
    if [ -d "$NNTRAINER_ROOT/builddir" ]; then
        rm -rf builddir
    fi
    ./tools/package_android.sh -Dmmap-read=false
else
    echo "nntrainer for Android already built."
fi

# Check if build was successful
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    echo "Error: nntrainer build failed. Please check the build logs."
    exit 1
fi

# Step 2: Check tokenizer library
echo "Checking Tokenizer Library"
cd "$SCRIPT_DIR"
mkdir -p lib
if [ ! -f "lib/libtokenizers_android_c.a" ]; then
    echo "libtokenizers_android_c.a not found in lib directory."
    echo ""
    echo "The tokenizer library needs to be built for Android arm64."
    echo "Attempting to build using build_tokenizer_android.sh..."
    echo ""
    if [ -f "build_tokenizer_android.sh" ]; then
        chmod +x build_tokenizer_android.sh
        ./build_tokenizer_android.sh arm64-v8a
        # The script should output to lib/libtokenizers_android_c.a
    else
        echo "Error: build_tokenizer_android.sh not found."
        echo "Please build the tokenizer library manually for Android arm64"
        echo "and place it in: $SCRIPT_DIR/lib/libtokenizers_android_c.a"
        exit 1
    fi
fi

# Verify tokenizer library is compatible
if [ -f "lib/libtokenizers_android_c.a" ]; then
    echo "Tokenizer Library Ready"
else
    echo "Error: Tokenizer library not available."
    exit 1
fi

# Step 3: Prepare json.hpp if not present
if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
    echo "json.hpp not found. Copying from CausalLM..."
    if [ -f "../CausalLM/json.hpp" ]; then
        cp ../CausalLM/json.hpp ./
    else
        echo "Error: Failed to get json.hpp"
        exit 1
    fi
fi

# Step 4: Build XLMRoberta application
echo "Building XLMRoberta application..."
cd "$SCRIPT_DIR/jni"

# Clean previous builds
rm -rf libs obj

# Run ndk-build
ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)

echo ""
echo "Build completed successfully!"
echo "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
echo ""
echo "Executables: nntr_xmlroberta, nntr_quantize"
echo "Libraries: libnntrainer.so, libccapi-nntrainer.so, libc++_shared.so"

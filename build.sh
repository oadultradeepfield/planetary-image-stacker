#!/bin/bash

# Stop the script if any command fails
set -e

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake configuration
cmake ..

# Compile the project using all available cores
make -j$(nproc)

echo "Build complete! Executables are in the build/ directory."

# Planetary Image Stacker

This project is a modification of my previous project, [Galilean](http://galilean.phanuphats.com/), which is also a planetary image stacking tool. The main difference is that this project removes the post-processing features and focuses solely on stacking images, while trying to make it super fast and memory efficient with C++. It also serves as an introduction for me to C++ and OpenMP.

## Requirements

- **C++17 Compiler** with OpenMP support (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake** 3.16.0+
- **OpenCV** 4.5.0+
- **OpenMP** (optional, for parallel processing)

### Installing Dependencies

#### Ubuntu/Debian

```bash
sudo apt install cmake build-essential libopencv-dev
```

#### macOS (with Homebrew)

```bash
brew install cmake opencv
brew install libomp
```

#### Windows

Install Visual Studio with C++ support, then use vcpkg:

```cmd
vcpkg install opencv
```

## Build

```bash
./build.sh
```

## Usage

### Processing Videos

```bash
./planetary_image_stacker <video_path> <crop_size> [frame_skip]
```

**Parameters:**

- `video_path`: Path to your planetary video file
- `crop_size`: Size of the crop in pixels (e.g., `640` for 640x640 crop around detected planet)
- `frame_skip (optional, default to 1)`: Number of frames to skip (e.g., `2` to use every 3rd frame)

**Example:**

```bash
./planetary_image_stacker jupiter_video.avi 480 2
```

### Running Tests

Process sample images and verify everything works:

```bash
./test-planetary_image_stacker
```

This processes test images in `test/input/` and saves stacked results to `test/output/`.

## How It Works

1. **Detection**: Automatically finds the planetary body in each frame/image
2. **Cropping**: Extracts a square region around the detected planet
3. **Alignment**: Aligns all cropped images to compensate for atmospheric movement
4. **Stacking**: Combines aligned images to reduce noise and enhance details

The result is a much sharper, cleaner planetary image than any single frame.

## Project Structure

```
├── src/           # Source code
├── include/       # Header files
├── test/
│   ├── input/     # Sample test images (Jupiter, Saturn, Moon)
│   └── output/    # Stacked results
├── build/         # Build files
└── build.sh       # Build script
```

## License

This project is licensed under the General Public License v3.0. See the [LICENSE](LICENSE) file for details.

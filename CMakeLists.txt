cmake_minimum_required(VERSION 3.16)
project(PlanetaryImageStacker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenMP)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
)

add_executable(planetary_image_stacker
    src/main.cpp
    src/image.cpp
    src/cropped_image.cpp
    src/video_processor.cpp
    src/planet_detector.cpp
    src/image_aligner.cpp
    src/image_stacker.cpp
)
target_link_libraries(planetary_image_stacker
    ${OpenCV_LIBS}
)

add_executable(test_planetary_image_stacker
    src/test.cpp
    src/image.cpp
    src/cropped_image.cpp
    src/planet_detector.cpp
    src/image_aligner.cpp
    src/image_stacker.cpp
)
target_link_libraries(test_planetary_image_stacker ${OpenCV_LIBS})

# Only link OpenMP if found
if(OpenMP_CXX_FOUND)
  target_link_libraries(planetary_image_stacker OpenMP::OpenMP_CXX)
  target_link_libraries(test_planetary_image_stacker OpenMP::OpenMP_CXX)
endif()

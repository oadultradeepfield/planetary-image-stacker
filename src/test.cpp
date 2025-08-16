#include "cropped_image.hpp"
#include "image.hpp"
#include "image_aligner.hpp"
#include "image_stacker.hpp"
#include "planet_detector.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

bool create_output_directory(const std::string &output_path) {
  fs::path dir = fs::path(output_path).parent_path();
  try {
    fs::create_directories(dir);
    return true;
  } catch (const fs::filesystem_error &e) {
    std::cerr << "Error creating directory: " << e.what() << std::endl;
    return false;
  }
}

bool process_image_set(const std::string &input_dir,
                       const std::string &output_path) {
  std::cout << "Processing directory: " << input_dir << std::endl;

  // Create output directory if it doesn't exist
  if (!create_output_directory(output_path)) {
    return false;
  }

  std::vector<CroppedImage> cropped_images;
  cropped_images.reserve(10);

  // Process images 1-10
  for (int i = 1; i <= 10; ++i) {
    std::string filename = input_dir + std::to_string(i) + ".png";

    try {
      std::cout << "  Processing: " << filename << std::endl;
      Image img(filename);
      auto cropped_img = PlanetDetector::crop(img, 480);
      cropped_images.push_back(cropped_img);
    } catch (const std::exception &e) {
      std::cerr << "  Error processing " << filename << ": " << e.what()
                << std::endl;
      return false;
    }
  }

  // Align images
  std::cout << "  Aligning images..." << std::endl;
  std::vector<cv::Mat> aligned_images =
      ImageAligner::align_images(cropped_images);

  // Stack images
  std::cout << "  Stacking images..." << std::endl;
  cv::Mat final_image = ImageStacker::stack_images(aligned_images);

  // Save the final result
  if (!cv::imwrite(output_path, final_image)) {
    std::cerr << "  Error saving image to: " << output_path << std::endl;
    return false;
  }

  std::cout << "  Successfully saved: " << output_path << std::endl;
  return true;
}

int test() {
  const std::vector<std::pair<std::string, std::string>> test_cases = {
      {"../test/input/jupiter_sample_frames/",
       "../test/output/jupiter_output.png"},
      {"../test/input/saturn_sample_frames/",
       "../test/output/saturn_output.png"},
      {"../test/input/moon_sample_frames/", "../test/output/moon_output.png"}};

  int successful_tests = 0;

  for (const auto &[input_dir, output_path] : test_cases) {
    if (process_image_set(input_dir, output_path)) {
      successful_tests++;
    } else {
      std::cerr << "Failed to process: " << input_dir << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "Completed " << successful_tests << "/" << test_cases.size()
            << " test cases successfully." << std::endl;

  return (successful_tests == test_cases.size()) ? 0 : 1;
}

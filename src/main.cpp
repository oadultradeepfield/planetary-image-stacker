#include "cropped_image.hpp"
#include "image_aligner.hpp"
#include "image_stacker.hpp"
#include "planet_detector.hpp"
#include "video_processor.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <vector>

namespace fs = std::filesystem;

int main(const int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
        << " <video_path> <crop_size> [frame_skip]\n";
    return 1;
  }

  std::string video_path = argv[1];
  int crop_size = std::stoi(argv[2]);
  int frame_skip = (argc > 3) ? std::stoi(argv[3]) : 1;

  std::cout << "Processing video: " << video_path
      << "\nCrop size: " << crop_size << "\nFrame skip: " << frame_skip
      << std::endl;

  // Prepare output directory
  fs::path input_path(video_path);
  fs::path output_dir = input_path.parent_path() / "output";
  fs::create_directories(output_dir);
  fs::path output_path =
      output_dir / (input_path.stem().string() + "_stacked.png");

  try {
    // Process video
    std::cout << "Step 1/3: Cropping frames..." << std::endl;
    std::vector<CroppedImage> cropped_images =
        VideoProcessor::processVideo(video_path, crop_size, frame_skip);
    std::cout << "  Cropped " << cropped_images.size() << " frames.\n";

    if (cropped_images.empty()) {
      std::cerr << "No images were cropped. Exiting." << std::endl;
      return 1;
    }

    // Align images
    std::cout << "Step 2/3: Aligning images..." << std::endl;
    std::vector<cv::Mat> aligned_images =
        ImageAligner::align_images(cropped_images);

    // Stack images
    std::cout << "Step 3/3: Stacking images..." << std::endl;
    cv::Mat final_image = ImageStacker::stack_images(aligned_images);

    // Save the final image
    if (!cv::imwrite(output_path.string(), final_image)) {
      std::cerr << "Error saving image to: " << output_path << std::endl;
      return 1;
    }

    std::cout << "Successfully saved stacked image to: " << output_path
        << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

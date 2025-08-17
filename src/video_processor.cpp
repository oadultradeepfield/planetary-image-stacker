#include "video_processor.hpp"
#include "cropped_image.hpp"
#include "planet_detector.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

std::vector<CroppedImage> VideoProcessor::processVideo(const std::string &video_path,
                                                       int crop_size,
                                                       int frame_skip) {
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("Could not open video file: " + video_path);
  }

  std::vector<cv::Mat> frames;
  cv::Mat frame;
  int frame_count = 0;

  // Step 1: Read frames sequentially
  while (cap.read(frame)) {
    if (frame_count % frame_skip == 0) {
      frames.push_back(frame.clone());
    }
    frame_count++;
  }

  // Step 2: Parallel cropping
  std::vector<CroppedImage> cropped_images;
  cropped_images.reserve(frames.size());

#pragma omp parallel for default(none) shared(frames, crop_size, cropped_images)
  for (const auto &f: frames) {
    Image image(f);
    CroppedImage cropped = PlanetDetector::crop(image, crop_size);
#pragma omp critical
    {
      cropped_images.push_back(cropped);
    }
  }

  return cropped_images;
}

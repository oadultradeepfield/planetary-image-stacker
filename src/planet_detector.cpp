#include "../include/planet_detector.hpp"

cv::Mat PlanetDetector::crop(const Image &image, int crop_size) {
  // Detect centroid first
  Centroid centroid = detect(image);

  int h = image.color.rows;
  int w = image.color.cols;

  // Adjust crop size to fit within image bounds
  int min_dimension = std::min(h, w);
  if (crop_size > min_dimension) {
    crop_size = min_dimension;
  }

  int half_crop = crop_size / 2;

  // Calculate source boundaries
  int src_x_min = std::max(centroid.x - half_crop, 0);
  int src_x_max = std::min(centroid.x + half_crop, w);
  int src_y_min = std::max(centroid.y - half_crop, 0);
  int src_y_max = std::min(centroid.y + half_crop, h);

  // Crop the region of interest
  cv::Rect crop_rect(src_x_min, src_y_min, src_x_max - src_x_min,
                     src_y_max - src_y_min);
  cv::Mat cropped_image = image.color(crop_rect);

  // Calculate the padding needed to make it pretty square
  int top = std::max(0, half_crop - centroid.y);
  int bottom = std::max(0, (h - centroid.y) - half_crop);
  int left = std::max(0, half_crop - centroid.x);
  int right = std::max(0, (w - centroid.x) - half_crop);

  // Add black padding to make the cropped image square
  cv::Mat padded_image;
  cv::copyMakeBorder(cropped_image, padded_image, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

  // Resize to ensure the output is exactly crop_size x crop_size
  cv::Mat result;
  cv::resize(padded_image, result, cv::Size(crop_size, crop_size));

  return result;
}

Centroid PlanetDetector::detect(const Image &image) {
  // Calculate moments of the binary image
  cv::Moments M = cv::moments(image.binary, true);

  if (M.m00 == 0) {
    throw std::runtime_error("No planet detected in the image.");
  }

  // Calculate the centroid
  int cx = static_cast<int>(M.m10 / M.m00);
  int cy = static_cast<int>(M.m01 / M.m00);

  return Centroid{cx, cy};
}

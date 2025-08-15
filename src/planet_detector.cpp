#include "planet_detector.hpp"
#include "cropped_image.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

CroppedImage PlanetDetector::crop(const Image &image, int crop_size) {
  // Detect centroid first
  Centroid centroid = detect(image);
  int h = image.get_color().rows;
  int w = image.get_color().cols;

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

  // Crop both color and grayscale images using the same rectangle
  cv::Mat cropped_color = image.get_color()(crop_rect);
  cv::Mat cropped_gray = image.get_grayscale()(crop_rect);

  // Calculate the padding needed to make it pretty square
  int top = std::max(0, half_crop - centroid.y);
  int bottom = std::max(0, (h - centroid.y) - half_crop);
  int left = std::max(0, half_crop - centroid.x);
  int right = std::max(0, (w - centroid.x) - half_crop);

  // Add black padding to both images cv::Mat padded_color, padded_gray;
  cv::Mat padded_color, padded_gray;
  cv::copyMakeBorder(cropped_color, padded_color, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  cv::copyMakeBorder(cropped_gray, padded_gray, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(0));

  // Resize both to ensure the output is exactly crop_size x crop_size
  cv::Mat result_color, result_gray;
  cv::resize(padded_color, result_color, cv::Size(crop_size, crop_size));
  cv::resize(padded_gray, result_gray, cv::Size(crop_size, crop_size));

  return CroppedImage(result_color, result_gray);
}

Centroid PlanetDetector::detect(const Image &image) {
  // Calculate moments of the binary image
  cv::Moments M = cv::moments(image.get_binary(), true);

  if (M.m00 == 0) {
    throw std::runtime_error("No planet detected in the image.");
  }

  // Calculate the centroid
  int cx = static_cast<int>(M.m10 / M.m00);
  int cy = static_cast<int>(M.m01 / M.m00);

  return Centroid{cx, cy};
}

#include "../include/image.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

Image::Image(const std::string &filename) {
  // Load the color image
  color = cv::imread(filename, cv::IMREAD_COLOR);

  if (color.empty()) {
    throw std::runtime_error("Could not open or find the image: " + filename);
  }

  // Generate grayscale and binary images
  generate_grayscale();
  generate_binary();
}

void Image::generate_grayscale() {
  cv::cvtColor(color, grayscale, cv::COLOR_BGR2GRAY);
}

void Image::generate_binary() {
  // Apply adaptive thresholding to convert grayscale to binary
  cv::Mat thresh;
  cv::adaptiveThreshold(grayscale, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY, 11, 2);

  // Apply morphological dilation to enhance the binary
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(thresh, binary, kernel, cv::Point(-1, -1), 2);
}

// get the quality of the image based on contrast, sharpness, and SNR
float Image::get_quality_score() {
  float contrast = get_contrast();
  float sharpness = get_sharpness();
  float snr = get_snr();
  return (0.2f * contrast + 0.5f * sharpness + 0.3f * snr);
}

// get contrast using standard deviation of pixel intensities
float Image::get_contrast() {
  cv::Scalar mean, stddev;
  cv::meanStdDev(grayscale, mean, stddev);
  return stddev[0];
}

// get sharpness using the Laplacian variance
float Image::get_sharpness() {
  cv::Mat laplacian;
  cv::Laplacian(grayscale, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  return stddev[0];
}

// get signal-to-noise ratio (SNR)
float Image::get_snr() {
  cv::Scalar mean, stddev;
  cv::meanStdDev(grayscale, mean, stddev);
  return mean[0] /
         (stddev[0] + 1e-8f); // small epsilon to avoid division by zero
}

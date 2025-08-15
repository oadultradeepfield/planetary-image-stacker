#include "image.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>

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

cv::Mat Image::get_color() const { return color; }
cv::Mat Image::get_grayscale() const { return grayscale; }
cv::Mat Image::get_binary() const { return binary; }

void Image::generate_grayscale() {
  if (grayscale.empty()) {
    cv::cvtColor(color, grayscale, cv::COLOR_BGR2GRAY);
  }
}

void Image::generate_binary() {
  // Generate grayscale if not already done
  generate_grayscale();

  // Apply adaptive thresholding to convert grayscale to binary
  if (binary.empty()) {
    cv::Mat thresh;
    cv::adaptiveThreshold(grayscale, thresh, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11,
                          2);

    // Apply morphological dilation to enhance the binary
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(thresh, binary, kernel, cv::Point(-1, -1), 2);
  }
}

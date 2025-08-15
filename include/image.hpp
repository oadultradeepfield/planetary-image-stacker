#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <opencv2/opencv.hpp>
#include <string>

class Image {
public:
  cv::Mat color;
  cv::Mat grayscale;
  cv::Mat binary;

  explicit Image(const std::string &filename);

private:
  void generate_grayscale();
  void generate_binary();
};

#endif

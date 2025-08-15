#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <opencv2/opencv.hpp>
#include <string>

class Image {
public:
  cv::Mat get_color() const { return color; }
  cv::Mat get_grayscale() const { return grayscale; }
  cv::Mat get_binary() const { return binary; }

  explicit Image(const std::string &filename);

protected:
  Image() = default;

  void generate_grayscale();
  void generate_binary();

  cv::Mat color;
  cv::Mat grayscale;
  cv::Mat binary;
};

#endif

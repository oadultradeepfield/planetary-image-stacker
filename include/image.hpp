#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <opencv2/core/mat.hpp>
#include <string>

class Image {
public:
  [[nodiscard]] cv::Mat get_color() const;

  [[nodiscard]] cv::Mat get_grayscale() const;

  [[nodiscard]] cv::Mat get_binary() const;

  explicit Image(const std::string &filename);

  explicit Image(const cv::Mat &img);

protected:
  Image() = default;

  void generate_grayscale();

  void generate_binary();

  cv::Mat color;
  cv::Mat grayscale;
  cv::Mat binary;
};

#endif

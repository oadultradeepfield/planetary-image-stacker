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
  float get_quality_score();

private:
  void generate_grayscale();
  void generate_binary();
  float get_contrast();
  float get_sharpness();
  float get_snr(); // signal-to-noise ratio
};

#endif

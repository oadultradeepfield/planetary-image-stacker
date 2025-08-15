#ifndef IMAGE_STACKER_HPP
#define IMAGE_STACKER_HPP

#include <opencv2/core/mat.hpp>
#include <vector>

class ImageStacker {
public:
  static cv::Mat stack_images(const std::vector<cv::Mat> &images);

private:
  // Private constructor to prevent instantiation
  ImageStacker() = default;
};

#endif

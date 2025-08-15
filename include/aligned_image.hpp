#ifndef ALIGNED_IMAGE_HPP
#define ALIGNED_IMAGE_HPP

#include "image.hpp"
#include <opencv2/opencv.hpp>

class AlignedImage : public Image {
public:
  AlignedImage(const cv::Mat &color_img);
};

#endif

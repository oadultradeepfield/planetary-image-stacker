#ifndef CROPPED_IMAGE_HPP
#define CROPPED_IMAGE_HPP

#include "image.hpp"

class CroppedImage : public Image {
public:
  CroppedImage(const cv::Mat &color_img, const cv::Mat &grayscale_img);

  float get_contrast();
  float get_sharpness();
  float get_snr();
};

#endif

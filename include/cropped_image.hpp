#ifndef CROPPED_IMAGE_HPP
#define CROPPED_IMAGE_HPP

#include "image.hpp"
#include <opencv2/core/mat.hpp>

class CroppedImage : public Image {
public:
  static float contrast_weight;
  static float sharpness_weight;
  static float snr_weight;

  CroppedImage(const cv::Mat &color_img, const cv::Mat &grayscale_img);

  float get_quality_score();
  bool operator<(const CroppedImage &other) const;

private:
  float quality_score;

  float get_contrast();
  float get_sharpness();
  float get_snr();
};

#endif

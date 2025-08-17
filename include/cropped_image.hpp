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

  [[nodiscard]] double get_quality_score() const;

  bool operator<(const CroppedImage &other) const;

private:
  double quality_score;

  [[nodiscard]] double get_contrast() const;

  [[nodiscard]] double get_sharpness() const;

  [[nodiscard]] double get_snr() const;
};

#endif

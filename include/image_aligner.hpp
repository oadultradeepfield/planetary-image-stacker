#ifndef IMAGE_ALIGNER_HPP
#define IMAGE_ALIGNER_HPP

#include "cropped_image.hpp"
#include <opencv2/core/mat.hpp>
#include <vector>

class ImageAligner {
public:
  static std::vector<cv::Mat> align_images(std::vector<CroppedImage> &images);

private:
  // Private constructor to prevent instantiation
  ImageAligner() = default;

  static cv::Point2d compute_phase_correlation(const cv::Mat &img1,
                                               const cv::Mat &img2);
  static CroppedImage select_template(const std::vector<CroppedImage> &images);
};

#endif

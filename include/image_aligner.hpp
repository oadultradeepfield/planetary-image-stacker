#ifndef IMAGE_ALIGNER_HPP
#define IMAGE_ALIGNER_HPP

#include "aligned_image.hpp"
#include "cropped_image.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

class ImageAligner {
public:
  static std::vector<AlignedImage>
  align_images(std::vector<CroppedImage> &images);

private:
  // Private constructor to prevent instantiation
  ImageAligner() = default;

  static CroppedImage select_template(const std::vector<CroppedImage> &images);
};

#endif

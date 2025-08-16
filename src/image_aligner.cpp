#include "image_aligner.hpp"
#include "cropped_image.hpp"
#include <algorithm>
#include <omp.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

std::vector<cv::Mat>
ImageAligner::align_images(std::vector<CroppedImage> &images) {
  if (images.empty())
    return {};

  CroppedImage template_image = select_template(images);
  cv::Mat template_gray = template_image.get_grayscale();

  std::vector<cv::Mat> aligned_images;
  aligned_images.resize(images.size());

  #pragma omp parallel for
  for (int i = 0; i < images.size(); ++i) {
    cv::Mat img = images[i].get_color();
    cv::Mat img_gray = images[i].get_grayscale();

    // Use phase correlation for sub-pixel accuracy
    cv::Point2d shift = compute_phase_correlation(img_gray, template_gray);

    // Apply translation
    cv::Mat aligned_img;
    cv::Mat translation_matrix =
        (cv::Mat_<double>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
    cv::warpAffine(img, aligned_img, translation_matrix, img.size());

    #pragma omp critical
    {
      aligned_images[i] = aligned_img;
    }
  }

  return aligned_images;
}

cv::Point2d ImageAligner::compute_phase_correlation(const cv::Mat &img1,
                                                    const cv::Mat &img2) {
  // Ensure images are the same size
  cv::Size common_size =
      cv::Size(std::min(img1.cols, img2.cols), std::min(img1.rows, img2.rows));

  cv::Mat src1, src2;
  cv::resize(img1, src1, common_size);
  cv::resize(img2, src2, common_size);

  // Convert to floating point
  src1.convertTo(src1, CV_32F);
  src2.convertTo(src2, CV_32F);

  // Apply window function to reduce edge effects
  cv::Mat window;
  cv::createHanningWindow(window, common_size, CV_32F);
  src1 = src1.mul(window);
  src2 = src2.mul(window);

  // Compute phase correlation
  cv::Point2d shift = cv::phaseCorrelate(src1, src2);

  return shift;
}

CroppedImage
ImageAligner::select_template(const std::vector<CroppedImage> &images) {
  auto max_it = std::max_element(images.begin(), images.end());
  return *max_it;
}

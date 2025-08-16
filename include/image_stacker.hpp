#ifndef IMAGE_STACKER_HPP
#define IMAGE_STACKER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class ImageStacker {
public:
  static float sigma_threshold; // kappa value for sigma clipping (default: 3.0)

  static cv::Mat stack_images(const std::vector<cv::Mat> &images);

private:
  static std::vector<cv::Mat>

  convert_to_float(const std::vector<cv::Mat> &images);

  static void compute_mean_and_std(const std::vector<cv::Mat> &float_images,
                                   cv::Mat &mean_img, cv::Mat &std_img);

  static cv::Mat compute_median(const std::vector<cv::Mat> &float_images);

  static cv::Mat
  apply_sigma_clipping_and_mean(const std::vector<cv::Mat> &float_images,
                                const cv::Mat &mean_img, const cv::Mat &std_img,
                                const cv::Mat &median_img);
};

#endif

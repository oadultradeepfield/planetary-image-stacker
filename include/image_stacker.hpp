#ifndef IMAGE_STACKER_HPP
#define IMAGE_STACKER_HPP

#include <opencv2/core/mat.hpp>
#include <vector>

class ImageStacker {
public:
  static float sigma_threshold;

  static cv::Mat stack_images(const std::vector<cv::Mat> &images);

private:
  static std::vector<cv::Mat>
  convert_to_float(const std::vector<cv::Mat> &images);

  // Helper functions for computing statistics
  static void compute_sum_and_square(const std::vector<const float *> &row_ptrs,
                                     int x, int ch, int channels,
                                     size_t num_images, double &sum,
                                     double &square_sum);

  static void compute_variance(double square_sum, double n, double mean,
                               double &variance);

  static void compute_clipped_sum(const std::vector<const float *> &row_ptrs,
                                  int x, int ch, int channels,
                                  size_t num_images, double mean,
                                  double threshold, double &clipped_sum,
                                  int &clipped_count);

  static void compute_out_value(const std::vector<const float *> &row_ptrs,
                                int x, int ch, size_t num_images, int channels,
                                double clipped_sum, int clipped_count,
                                float &out_val);

  // Private constructor to prevent instantiation
  ImageStacker() = default;
};

#endif

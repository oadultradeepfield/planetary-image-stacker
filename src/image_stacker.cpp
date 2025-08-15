#include "image_stacker.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

float ImageStacker::sigma_threshold = 2.0f;

cv::Mat ImageStacker::stack_images(const std::vector<cv::Mat> &images) {
  if (images.empty()) {
    throw std::invalid_argument("No images provided for stacking.");
  }

  const cv::Size img_size = images[0].size();
  const int img_type = images[0].type();
  const int channels = images[0].channels();
  const size_t num_images = images.size();

  // Result as float with same channel count
  std::vector<cv::Mat> float_images = convert_to_float(images);
  cv::Mat result(img_size, CV_MAKETYPE(CV_32F, channels), cv::Scalar::all(0));

  // Parallelize over pixels;
  // collapse(2) gives independent iterations per (y,x).
  // Use schedule(static) to avoid runtime overhead of dynamic chunking.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < img_size.height; ++y) {
    for (int x = 0; x < img_size.width; ++x) {

      std::vector<const float *> row_ptrs;
      row_ptrs.resize(num_images);
      for (size_t k = 0; k < num_images; ++k) {
        row_ptrs[k] = float_images[k].ptr<float>(y);
      }

      // For each channel independently
      for (int ch = 0; ch < channels; ++ch) {
        double sum;
        double square_sum;
        compute_sum_and_square(row_ptrs, x, ch, channels, num_images, sum,
                               square_sum);

        const double n = static_cast<double>(num_images);
        const double mean = sum / n;

        double var;
        compute_variance(square_sum, n, mean, var);
        const double stddev = std::sqrt(std::max(0.0, var));

        double clipped_sum;
        int clipped_count;
        compute_clipped_sum(row_ptrs, x, ch, channels, num_images, mean,
                            sigma_threshold * stddev, clipped_sum,
                            clipped_count);

        float out_val;
        compute_out_value(row_ptrs, x, ch, num_images, channels, clipped_sum,
                          clipped_count, out_val);

        // write into result (interleaved channels)
        result.ptr<float>(y)[x * channels + ch] = out_val;
      } // channel loop
    } // x
  } // y

  // Convert result back to original type
  cv::Mat finalResult;
  result.convertTo(finalResult, img_type);
  return finalResult;
}

std::vector<cv::Mat>
ImageStacker::convert_to_float(const std::vector<cv::Mat> &images) {
  const size_t num_images = images.size();

  std::vector<cv::Mat> float_images;
  float_images.reserve(num_images);
  for (size_t i = 0; i < num_images; ++i) {
    cv::Mat tmp;
    images[i].convertTo(tmp, CV_32F); // preserves channels
    float_images.push_back(std::move(tmp));
  }

  return float_images;
}

// Helper functions for computing statistics

// Computes the sum and sum of squares for a given pixel/channel across all
// images
void ImageStacker::compute_sum_and_square(
    const std::vector<const float *> &row_ptrs, int x, int ch, int channels,
    size_t num_images, double &sum, double &square_sum) {
  sum = 0.0;
  square_sum = 0.0;
  for (size_t k = 0; k < num_images; ++k) {
    float v = row_ptrs[k][x * channels + ch];
    sum += v;
    square_sum += static_cast<double>(v) * v;
  }
}

// Computes variance given sum of squares, count, and mean
void ImageStacker::compute_variance(double square_sum, double n, double mean,
                                    double &variance) {
  variance = (square_sum / n) - (mean * mean);
  if (variance < 0.0 && variance > -1e-12)
    variance = 0.0; // numerical safety
}

// Computes the sum and count of values within a threshold of the mean
void ImageStacker::compute_clipped_sum(
    const std::vector<const float *> &row_ptrs, int x, int ch, int channels,
    size_t num_images, double mean, double threshold, double &clipped_sum,
    int &clipped_count) {
  clipped_sum = 0.0;
  clipped_count = 0;
  for (size_t k = 0; k < num_images; ++k) {
    float v = row_ptrs[k][x * channels + ch];
    if (std::fabs(v - mean) <= threshold) {
      clipped_sum += v;
      ++clipped_count;
    }
  }
}

// Computes the output value for a pixel/channel, using mean if clipped values
// exist, otherwise median
void ImageStacker::compute_out_value(const std::vector<const float *> &row_ptrs,
                                     int x, int ch, size_t num_images,
                                     int channels, double clipped_sum,
                                     int clipped_count, float &out_val) {
  out_val = 0.0f;
  if (clipped_count > 0) {
    out_val = static_cast<float>(clipped_sum / clipped_count);
  } else {
    std::vector<float> vals;
    vals.reserve(num_images);
    for (size_t k = 0; k < num_images; ++k) {
      vals.push_back(row_ptrs[k][x * channels + ch]);
    }
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    out_val = vals[vals.size() / 2];
  }
}

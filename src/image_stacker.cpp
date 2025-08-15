#include "image_stacker.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

cv::Mat ImageStacker::stack_images(const std::vector<cv::Mat> &images) {
  if (images.empty()) {
    throw std::invalid_argument("No images provided for stacking.");
  }

  const cv::Size img_size = images[0].size();
  const int img_type = images[0].type();
  const int channels = images[0].channels();
  const size_t num_images = images.size();

  // Convert all to float (same number of channels)
  std::vector<cv::Mat> float_images;
  float_images.reserve(num_images);
  for (size_t i = 0; i < num_images; ++i) {
    cv::Mat tmp;
    images[i].convertTo(tmp, CV_32F); // preserves channels
    float_images.push_back(std::move(tmp));
  }

  // Result as float with same channel count
  cv::Mat result(img_size, CV_MAKETYPE(CV_32F, channels), cv::Scalar::all(0));

  const float sigma_threshold = 2.0f;

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
        double sum = 0.0;
        double square_sum = 0.0;
        for (size_t k = 0; k < num_images; ++k) {
          float v = row_ptrs[k][x * channels + ch];
          sum += v;
          square_sum += static_cast<double>(v) * v;
        }

        const double n = static_cast<double>(num_images);
        const double mean = sum / n;

        double var = (square_sum / n) - (mean * mean);
        if (var < 0.0 && var > -1e-12)
          var = 0.0; // numerical safety
        const double stddev = std::sqrt(std::max(0.0, var));

        double clipped_sum = 0.0;
        int clipped_count = 0;
        const double threshold = sigma_threshold * stddev;
        if (stddev == 0.0) {
          // if stddev == 0, all values are identical -> take mean
          clipped_sum = sum;
          clipped_count = static_cast<int>(num_images);
        } else {
          for (size_t k = 0; k < num_images; ++k) {
            float v = row_ptrs[k][x * channels + ch];
            if (std::fabs(v - mean) <= threshold) {
              clipped_sum += v;
              ++clipped_count;
            }
          }
        }

        float out_val = 0.0f;
        if (clipped_count > 0) {
          out_val = static_cast<float>(clipped_sum / clipped_count);
        } else {
          // fallback: compute median
          std::vector<float> vals;
          vals.reserve(num_images);
          for (size_t k = 0; k < num_images; ++k) {
            vals.push_back(row_ptrs[k][x * channels + ch]);
          }
          std::nth_element(vals.begin(), vals.begin() + vals.size() / 2,
                           vals.end());
          out_val = vals[vals.size() / 2];
        }

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

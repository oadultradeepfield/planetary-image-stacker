#include "image_stacker.hpp"
#include <algorithm>
#include <omp.h>
#include <opencv2/core/mat.hpp>
#include <stdexcept>
#include <vector>

float ImageStacker::sigma_threshold = 3.0f;

cv::Mat ImageStacker::stack_images(const std::vector<cv::Mat> &images) {
  if (images.empty()) {
    throw std::invalid_argument("No images provided for stacking.");
  }

  // Validate all images have same dimensions and type
  const cv::Size img_size = images[0].size();
  const int img_type = images[0].type();
  const size_t num_images = images.size();

  for (size_t i = 1; i < num_images; ++i) {
    if (images[i].size() != img_size || images[i].type() != img_type) {
      throw std::invalid_argument(
        "All images must have same dimensions and type.");
    }
  }

  // Convert all images to float for processing
  std::vector<cv::Mat> float_images = convert_to_float(images);

  // Pre-compute mean and standard deviation images
  cv::Mat mean_img, std_img;
  compute_mean_and_std(float_images, mean_img, std_img);

  // Compute median image
  const cv::Mat median_img = compute_median(float_images);

  // Apply sigma clipping and compute final mean
  const cv::Mat result = apply_sigma_clipping_and_mean(float_images, mean_img,
                                                       std_img, median_img);

  // Convert result back to original type
  cv::Mat final_result;
  result.convertTo(final_result, img_type);
  return final_result;
}

std::vector<cv::Mat>
ImageStacker::convert_to_float(const std::vector<cv::Mat> &images) {
  std::vector<cv::Mat> float_images;
  float_images.reserve(images.size());

  for (const auto &img: images) {
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F);
    float_images.push_back(std::move(float_img));
  }

  return float_images;
}

void ImageStacker::compute_mean_and_std(
  const std::vector<cv::Mat> &float_images, cv::Mat &mean_img,
  cv::Mat &std_img) {
  if (float_images.empty())
    return;

  const cv::Size img_size = float_images[0].size();
  const int channels = float_images[0].channels();
  const size_t num_images = float_images.size();

  // Initialize accumulator matrices
  mean_img = cv::Mat::zeros(img_size, CV_MAKETYPE(CV_32F, channels));
  cv::Mat sum_sq = cv::Mat::zeros(img_size, CV_MAKETYPE(CV_32F, channels));

  // Accumulate sum and sum of squares
  for (const auto &img: float_images) {
    mean_img += img;
    cv::Mat img_sq;
    cv::multiply(img, img, img_sq);
    sum_sq += img_sq;
  }

  // Compute mean
  mean_img /= static_cast<float>(num_images);

  // Compute variance
  cv::Mat mean_sq;
  cv::multiply(mean_img, mean_img, mean_sq);
  const cv::Mat variance = sum_sq / static_cast<float>(num_images) - mean_sq;

  // Compute standard deviation
  cv::sqrt(variance, std_img);
}

cv::Mat ImageStacker::compute_median(const std::vector<cv::Mat> &float_images) {
  if (float_images.empty())
    return {};

  const cv::Size img_size = float_images[0].size();
  const int channels = float_images[0].channels();
  const size_t num_images = float_images.size();

  cv::Mat median_img(img_size, CV_MAKETYPE(CV_32F, channels));

  // Parallelize over pixels
#pragma omp parallel for default(none) collapse(2) schedule(static) \
  shared(img_size, channels, median_img, num_images, float_images)
  for (int y = 0; y < img_size.height; ++y) {
    for (int x = 0; x < img_size.width; ++x) {
      for (int ch = 0; ch < channels; ++ch) {
        // Collect values for this pixel/channel
        std::vector<float> values;
        values.reserve(num_images);

        for (size_t i = 0; i < num_images; ++i) {
          values.push_back(float_images[i].ptr<float>(y)[x * channels + ch]);
        }

        // Find median using nth_element
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
        float median_val = values[mid];

        // For even number of elements, average the two middle values
        if (values.size() % 2 == 0 && values.size() > 1) {
          auto max_it = std::max_element(values.begin(), values.begin() + static_cast<long>(mid));
          median_val = (median_val + *max_it) * 0.5f;
        }

        median_img.ptr<float>(y)[x * channels + ch] = median_val;
      }
    }
  }

  return median_img;
}

cv::Mat ImageStacker::apply_sigma_clipping_and_mean(
  const std::vector<cv::Mat> &float_images, const cv::Mat &mean_img,
  const cv::Mat &std_img, const cv::Mat &median_img) {
  if (float_images.empty())
    return {};

  const cv::Size img_size = float_images[0].size();
  const int channels = float_images[0].channels();
  const size_t num_images = float_images.size();

  cv::Mat result = cv::Mat::zeros(img_size, CV_MAKETYPE(CV_32F, channels));

  // Parallelize over pixels
#pragma omp parallel for default(none) collapse(2) schedule(static) \
  shared(result, img_size, channels, mean_img, std_img, median_img, float_images, num_images)
  for (int y = 0; y < img_size.height; ++y) {
    for (int x = 0; x < img_size.width; ++x) {
      for (int ch = 0; ch < channels; ++ch) {
        const int pixel_idx = x * channels + ch;
        const float mean_val = mean_img.ptr<float>(y)[pixel_idx];
        const float std_val = std_img.ptr<float>(y)[pixel_idx];
        const float median_val = median_img.ptr<float>(y)[pixel_idx];
        const float threshold = sigma_threshold * std_val;

        double sum = 0.0;
        int count = 0;

        // Apply sigma clipping: replace outliers with median, then compute mean
        for (size_t i = 0; i < num_images; ++i) {
          float pixel_val = float_images[i].ptr<float>(y)[pixel_idx];

          // Check if pixel is within sigma threshold
          if (std::abs(pixel_val - mean_val) > threshold) {
            pixel_val = median_val;
          }

          sum += pixel_val;
          ++count;
        }

        result.ptr<float>(y)[pixel_idx] = static_cast<float>(sum / count);
      }
    }
  }

  return result;
}

#include "cropped_image.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

// Static member initialization
float CroppedImage::contrast_weight = 0.2f;
float CroppedImage::sharpness_weight = 0.5f;
float CroppedImage::snr_weight = 0.3f;

CroppedImage::CroppedImage(const cv::Mat &color_img,
                           const cv::Mat &grayscale_img)
  : Image() {
  color = color_img;
  grayscale = grayscale_img;
  quality_score = contrast_weight * get_contrast() +
                  sharpness_weight * get_sharpness() + snr_weight * get_snr();
}

double CroppedImage::get_quality_score() const { return quality_score; }

bool CroppedImage::operator<(const CroppedImage &other) const {
  return quality_score < other.quality_score;
}

// get contrast using standard deviation of pixel intensities
double CroppedImage::get_contrast() const {
  cv::Scalar mean, stddev;
  cv::meanStdDev(grayscale, mean, stddev);
  return stddev[0];
}

// get sharpness using the Laplacian variance
double CroppedImage::get_sharpness() const {
  cv::Mat laplacian;
  cv::Laplacian(grayscale, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  return stddev[0];
}

// get signal-to-noise ratio (SNR)
double CroppedImage::get_snr() const {
  cv::Scalar mean, stddev;
  cv::meanStdDev(grayscale, mean, stddev);
  return mean[0] /
         (stddev[0] + 1e-8f); // small epsilon to avoid division by zero
}

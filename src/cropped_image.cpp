#include "cropped_image.hpp"

CroppedImage::CroppedImage(const cv::Mat &color_img,
                           const cv::Mat &grayscale_img)
    : Image("") {
  color = color_img;
  grayscale = grayscale_img;
}

// get contrast using standard deviation of pixel intensities
float CroppedImage::get_contrast() {
  cv::Scalar mean, stddev;
  cv::meanStdDev(grayscale, mean, stddev);
  return stddev[0];
}

// get sharpness using the Laplacian variance
float CroppedImage::get_sharpness() {
  cv::Mat laplacian;
  cv::Laplacian(grayscale, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  return stddev[0];
}

// get signal-to-noise ratio (SNR)
float CroppedImage::get_snr() {
  cv::Scalar mean, stddev;
  cv::meanStdDev(grayscale, mean, stddev);
  return mean[0] /
         (stddev[0] + 1e-8f); // small epsilon to avoid division by zero
}

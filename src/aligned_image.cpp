
#include "aligned_image.hpp"
#include <opencv2/opencv.hpp>

AlignedImage::AlignedImage(const cv::Mat &color_img) : Image() {
  color = color_img;
}

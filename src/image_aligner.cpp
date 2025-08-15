#include "image_aligner.hpp"
#include "cropped_image.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::Mat>
ImageAligner::align_images(std::vector<CroppedImage> &images) {
  cv::Mat template_gray = select_template(images).get_grayscale();

  std::vector<cv::Mat> aligned_images;

  for (auto &image : images) {
    cv::Mat img = image.get_color();
    cv::Mat img_gray = image.get_grayscale();

    // Perform phase correlation to find translation
    cv::Mat result;
    cv::matchTemplate(img_gray, template_gray, result, cv::TM_CCOEFF_NORMED);
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

    // Compute the translation vector
    cv::Point translation =
        max_loc - cv::Point(template_gray.cols / 2, template_gray.rows / 2);

    // Align the image
    cv::Mat aligned_img;
    cv::Mat translation_matrix =
        (cv::Mat_<double>(2, 3) << 1, 0, translation.x, 0, 1, translation.y);
    cv::warpAffine(img, aligned_img, translation_matrix, img.size());

    // Store the aligned image
    aligned_images.push_back(aligned_img);
  }

  return aligned_images;
}

CroppedImage
ImageAligner::select_template(const std::vector<CroppedImage> &images) {
  auto max_it = std::max_element(images.begin(), images.end());
  return *max_it;
}

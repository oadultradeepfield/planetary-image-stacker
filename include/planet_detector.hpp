#ifndef PLANET_DETECTOR_HPP
#define PLANET_DETECTOR_HPP

#include "image.hpp"
#include <opencv2/opencv.hpp>

struct Centroid {
  int x;
  int y;
};

class PlanetDetector {
public:
  static cv::Mat crop(const Image &image, int crop_size);

private:
  // Private constructor to prevent instantiation
  PlanetDetector() = default;

  static Centroid detect(const Image &image);
};

#endif

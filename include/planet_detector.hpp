#ifndef PLANET_DETECTOR_HPP
#define PLANET_DETECTOR_HPP

#include "cropped_image.hpp"
#include "image.hpp"

struct Centroid {
  int x;
  int y;
};

class PlanetDetector {
public:
  static CroppedImage crop(const Image &image, int crop_size);

private:
  // Private constructor to prevent instantiation
  PlanetDetector() = default;

  static Centroid detect(const Image &image);
};

#endif

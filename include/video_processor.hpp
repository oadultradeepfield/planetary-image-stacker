#ifndef VIDEO_PROCESSOR_HPP
#define VIDEO_PROCESSOR_HPP

#include "cropped_image.hpp"
#include <opencv2/videoio.hpp>
#include <vector>

class VideoProcessor {
public:
    static std::vector<CroppedImage>
    processVideo(const std::string &video_path, int crop_size, int frame_skip = 1);

private:
    // Private constructor to prevent instantiation
    VideoProcessor() = default;
};

#endif
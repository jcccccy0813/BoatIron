#ifndef STEREO_MATCHING_HPP
#define STEREO_MATCHING_HPP

#include "Common.h"
#include <string>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

class StereoMatching {
public:
    // 构造函数改为无参数
    StereoMatching();

    // 主处理函数
    bool process(
        const std::string& imageList = "stereo_pairs.txt",
        const std::string& intrinsicFile = "intrinsics.yml",
        const std::string& extrinsicFile = "extrinsics.yml",
        const std::string& disparityOutput = "disparity",
        const std::string& pointCloudOutput = "pointcloud",
        StereoAlgorithm algorithm = STEREO_SGBM,
        int maxDisparity = 176,
        bool colorDisplay = true
    );

    // 获取帮助信息
    static void printHelp();
    static void stereoMatching();

private:
    struct Parameters {
        std::string imageList;
        std::string intrinsicFile;
        std::string extrinsicFile;
        std::string disparityOutput;
        std::string pointCloudOutput;
        StereoAlgorithm algorithm;
        int maxDisparity;
        bool colorDisplay;
    };

    void initStereoBM(cv::Ptr<cv::StereoBM>& bm, int maxDisparity);
    void initStereoSGBM(cv::Ptr<cv::StereoSGBM>& sgbm, int maxDisparity, int channels);
    void saveColoredXYZ(const std::string& filename, const cv::Mat& mat, const cv::Mat& color_img);

    Parameters params_;
};

#endif // STEREO_MATCHING_HPP
#pragma once
#include "Common.h"
#include <string>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <ctime>

class StereoCalibration {
public:
    // 预设配置枚举
    enum PresetType {
        PRESET_CHESSBOARD_11X8_20MM,
        PRESET_CHARUCO_9X6_25MM
    };

    // 使用预设配置进行双目标定
    bool calibrateWithPreset(const std::string& imageListFile,
        const std::string& leftIntrinsicFile = "left_camera.yml",
        const std::string& rightIntrinsicFile = "right_camera.yml",
        PresetType preset = PRESET_CHESSBOARD_11X8_20MM);

    // 原始标定方法
    bool calibrate(const std::vector<cv::String>& imagelist,
        cv::Size boardSize, PatternType pattern,
        float squareSize, float markerSize,
        cv::aruco::PredefinedDictionaryType arucoDict,
        const std::string& leftIntrinsicFile,
        const std::string& rightIntrinsicFile,
        bool displayCorners = false, bool showRectified = true);
    // 从文件中读取图像列表
    bool readStringList(const std::string& filename, std::vector<cv::String>& l);
    // 获取双目参数
    const StereoParameters& getStereoParams() const { return stereoParams_; }
    // 双目标定
    static void stereoCalibration();

private:
    struct PresetConfig {
        cv::Size boardSize;
        PatternType pattern;
        float squareSize;
        float markerSize;
        cv::aruco::PredefinedDictionaryType arucoDict;
    };

    PresetConfig getPresetConfig(PresetType preset);
    bool saveCalibrationResults(const std::string& intrinsicsFile,
        const std::string& extrinsicsFile);

    StereoParameters stereoParams_;
    cv::Mat cameraMatrix[2], distCoeffs[2];
};
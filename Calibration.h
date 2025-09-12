#ifndef CALIBRATION_HPP
#define CALIBRATION_HPP


#include <string>
#include <vector>
#include <iostream>
#include <ctime>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include "Common.h"

class Calibration {
public:
    // 构造函数
    Calibration();
    // 析构函数
    struct PresetConfig {
        cv::Size boardSize;
        PatternType pattern;
        float squareSize;
        float markerSize;
        cv::aruco::PredefinedDictionaryType arucoDict;
    };
    // 预设类型
    enum PresetType {
        PRESET_CHESSBOARD_11X8_20MM,
        PRESET_CHARUCO_9X6_25MM
    };
    // 单相机标定
    bool calibrateSingleCamera(const std::vector<std::string>& imageList,
        cv::Size boardSize, PatternType pattern,
        float squareSize, float markerSize,
        cv::aruco::PredefinedDictionaryType arucoDict,
        const std::string& outputFile);
    // 获取相机参数
    const CameraParameters& getCameraParams() const { return cameraParams_; }
    // 多相机标定
    bool readStringList(const std::string& filename, std::vector<std::string>& l);
    // 多相机标定，基于预设参数
    bool calibrateWithPreset(const std::string& imageListFile,
        const std::string& outputFile,
        PresetType preset = PRESET_CHESSBOARD_11X8_20MM);
    // 相机类型
    enum CameraType {
        LEFT_CAMERA,
        RIGHT_CAMERA
    };
    // 多相机标定
    bool calibrateSelectedCamera(CameraType cameraType, PresetType preset = PRESET_CHESSBOARD_11X8_20MM);
    // 单左相机标定
    static void  calibrateLeftCamera();
    // 单右相机标定
    static void  calibrateRightCamera();
private:
    bool runCalibration(const std::vector<std::vector<cv::Point2f>>& imagePoints,
        cv::Size imageSize, cv::Size boardSize, PatternType pattern,
        float squareSize, float aspectRatio, float gridWidth,
        bool releaseObject, int flags, cv::Mat& cameraMatrix,
        cv::Mat& distCoeffs, std::vector<cv::Mat>& rvecs,
        std::vector<cv::Mat>& tvecs, std::vector<float>& reprojErrs,
        std::vector<cv::Point3f>& newObjPoints, double& totalAvgErr);

    void calcChessboardCorners(cv::Size boardSize, float squareSize,
        std::vector<cv::Point3f>& corners, PatternType patternType);

    double computeReprojectionErrors(const std::vector<std::vector<cv::Point3f>>& objectPoints,
        const std::vector<std::vector<cv::Point2f>>& imagePoints,
        const std::vector<cv::Mat>& rvecs,
        const std::vector<cv::Mat>& tvecs,
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs,
        std::vector<float>& perViewErrors);

    // 获取预设配置
    PresetConfig getPresetConfig(PresetType preset);

    CameraParameters cameraParams_;

};

#endif // CALIBRATION_HPP
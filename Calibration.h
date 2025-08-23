#ifndef CALIBRATION_HPP
#define CALIBRATION_HPP

#include "Common.h"
#include <string>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <iostream>
#include <ctime>

class Calibration {
public:
    Calibration();
    struct PresetConfig {
        cv::Size boardSize;
        PatternType pattern;
        float squareSize;
        float markerSize;
        cv::aruco::PredefinedDictionaryType arucoDict;
    };
    enum PresetType {
        PRESET_CHESSBOARD_11X8_20MM,
        PRESET_CHARUCO_9X6_25MM
    };
    bool calibrateSingleCamera(const std::vector<std::string>& imageList,
        cv::Size boardSize, PatternType pattern,
        float squareSize, float markerSize,
        cv::aruco::PredefinedDictionaryType arucoDict,
        const std::string& outputFile);

    const CameraParameters& getCameraParams() const { return cameraParams_; }

    bool readStringList(const std::string& filename, std::vector<std::string>& l);
    bool calibrateWithPreset(const std::string& imageListFile,
        const std::string& outputFile,
        PresetType preset = PRESET_CHESSBOARD_11X8_20MM);
    enum CameraType {
        LEFT_CAMERA,
        RIGHT_CAMERA
    };

    bool calibrateSelectedCamera(CameraType cameraType, PresetType preset = PRESET_CHESSBOARD_11X8_20MM);
    static void  calibrateLeftCamera();
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

    // ªÒ»°‘§…Ë≈‰÷√
    PresetConfig getPresetConfig(PresetType preset);

    CameraParameters cameraParams_;

};

#endif // CALIBRATION_HPP
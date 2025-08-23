#ifndef COMMON_HPP
#define COMMON_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <limits>

enum PatternType {
    CHESSBOARD,
    CHARUCOBOARD,
    CIRCLES_GRID,
    ASYMMETRIC_CIRCLES_GRID
};

enum StereoAlgorithm {
    STEREO_BM,
    STEREO_SGBM,
    STEREO_HH,
    STEREO_HH4,
    STEREO_3WAY
};

struct CameraParameters {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Size imageSize;
    };

struct StereoParameters {
    cv::Mat R, T, E, F;
    cv::Mat R1, R2, P1, P2, Q;
    };
static void printMainMenu() {
    std::cout << "\n=== Stereo Vision System ===\n";
    std::cout << "1. Camera Capture\n";
    std::cout << "2. Structured Light Capture (Auto)\n";
    std::cout << "3. Calibrate Left Camera\n";
    std::cout << "4. Calibrate Right Camera\n";
    std::cout << "5. Stereo Calibration\n";
    std::cout << "6. Stereo Matching\n";
    std::cout << "7. Generate Gray Code Patterns\n";
    std::cout << "8. Decode Gray Code Patterns\n";
    std::cout << "9. Generate Image List\n";
    std::cout << "10. Exit\n";
    std::cout << "Enter your choice (1-10): ";
};
static void clearInputBuffer() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

#endif // COMMON_HPP#pragma once

#ifndef GRAY_CODE_DECODER_H
#define GRAY_CODE_DECODER_H

#include <opencv2/opencv.hpp>
#include <opencv2/structured_light.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include"Common.h"

class GrayCodeDecoder {
public:
    enum CameraSide {
        LEFT_CAMERA,
        RIGHT_CAMERA
    };

    struct Params {
        int width = 860;
        int height = 573;
        int whiteThreshold = 5;
        int blackThreshold = 45;
        std::string maskOutput = "mask.png";
        std::string xPngOutput = "x.png";
        std::string yPngOutput = "y.png";
        std::string xExrOutput = "x.exr";
        std::string yExrOutput = "y.exr";
    };

    GrayCodeDecoder();
    explicit GrayCodeDecoder(CameraSide side);
    explicit GrayCodeDecoder(const Params& params, CameraSide side = LEFT_CAMERA);

    // 设置相机侧别
    void setCameraSide(CameraSide side);
    CameraSide getCameraSide() const { return cameraSide_; }

    // 设置参数
    void setParams(const Params& params);
    Params getParams() const { return params_; }

    // 解码灰度码
    bool decode();
    bool decode(const std::string& imageListFile);
    bool decode(const std::vector<std::string>& imageFiles);

    // 获取解码结果
    const cv::Mat& getDecodedImage() const { return decodedImage_; }
    const cv::Mat& getShadowMask() const { return shadowMask_; }
    const cv::Mat& getVisualizationMask() const { return visualizationMask_; }

    // 保存结果
    bool saveResults();
    static void decodeGrayCodePatterns();

private:
    // 工具函数
    cv::Mat computeShadowMask(const cv::Mat& blackImage, const cv::Mat& whiteImage);
    cv::Mat computeDecodeImage(const std::vector<cv::Mat>& capturedPattern, const cv::Mat& mask);
    cv::Mat getDecodedMask(const cv::Mat& decoded);
    void visualizeDecodedImage(const cv::Mat& decoded, const std::string& xPath, const std::string& yPath);
    void saveDecodedImage(const cv::Mat& decoded, const std::string& xPath, const std::string& yPath);

    std::vector<std::string> loadImageList(const std::string& filename);
    std::vector<cv::Mat> loadImages(const std::vector<std::string>& filenames);

    // 根据相机侧别设置参数
    void setupCameraSpecificParams();

    Params params_;
    CameraSide cameraSide_;
    cv::Ptr<cv::structured_light::GrayCodePattern> graycode_;
    cv::Mat decodedImage_;
    cv::Mat shadowMask_;
    cv::Mat visualizationMask_;

    bool initializeGrayCode();
};

#endif // GRAY_CODE_DECODER_H
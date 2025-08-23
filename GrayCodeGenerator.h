#ifndef GRAY_CODE_GENERATOR_H
#define GRAY_CODE_GENERATOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/structured_light.hpp>
#include <string>
#include <vector>
#include <iostream>

class GrayCodeGenerator {
public:
    struct Params {
        int width = 512;
        int height = 384;
        std::string outputPath = ".";
    };

    GrayCodeGenerator();
    explicit GrayCodeGenerator(const Params& params);

    // 设置参数
    void setParams(const Params& params);
    Params getParams() const { return params_; }

    // 生成灰度码模式
    bool generatePatterns();

    // 获取生成的信息
    size_t getPatternCount() const { return patterns_.size(); }
    const std::vector<cv::Mat>& getPatterns() const { return patterns_; }

    // 保存模式到文件
    bool savePatterns(const std::string& customPath = "");

    // 获取黑白图像（用于阴影掩码计算）
    void getShadowMaskImages(cv::Mat& black, cv::Mat& white) const;
    static void generateGrayCodePatterns();

private:
    Params params_;
    cv::Ptr<cv::structured_light::GrayCodePattern> graycode_;
    std::vector<cv::Mat> patterns_;
    cv::Mat whiteImage_;
    cv::Mat blackImage_;

    bool initializeGrayCode();
};

#endif // GRAY_CODE_GENERATOR_H
#include "GrayCodeGenerator.h"


GrayCodeGenerator::GrayCodeGenerator() {
    params_.width = 860;
    params_.height = 573;
    params_.outputPath = ".";
    initializeGrayCode();
}
// 构造函数
GrayCodeGenerator::GrayCodeGenerator(const Params& params)
    : params_(params) {
    initializeGrayCode();
}
// 设置参数
void GrayCodeGenerator::setParams(const Params& params) {
    params_ = params;
    initializeGrayCode();
}
// 初始化
bool GrayCodeGenerator::initializeGrayCode() {
    cv::structured_light::GrayCodePattern::Params gcParams;
    gcParams.width = params_.width;
    gcParams.height = params_.height;

    graycode_ = cv::structured_light::GrayCodePattern::create(gcParams);
    return !graycode_.empty();
}
// 生成灰度码模式
bool GrayCodeGenerator::generatePatterns() {
    if (graycode_.empty()) {
        if (!initializeGrayCode()) {
            std::cerr << "Failed to initialize GrayCode pattern generator!" << std::endl;
            return false;
        }
    }

    patterns_.clear();
    graycode_->generate(patterns_);

    // 获取黑白图像用于阴影掩码计算
    graycode_->getImagesForShadowMasks(blackImage_, whiteImage_);
    patterns_.push_back(whiteImage_);
    patterns_.push_back(blackImage_);

    return !patterns_.empty();
}
// 保存模式
bool GrayCodeGenerator::savePatterns(const std::string& customPath) {
    if (patterns_.empty()) {
        std::cerr << "No patterns to save. Please generate patterns first." << std::endl;
        return false;
    }

    std::string savePath = customPath.empty() ? params_.outputPath : customPath;

    for (size_t i = 0; i < patterns_.size(); ++i) {
        std::string filename = savePath + "/pattern_" +
            (i < 10 ? "0" : "") + std::to_string(i + 1) + ".png";
        if (!cv::imwrite(filename, patterns_[i])) {
            std::cerr << "Failed to save pattern: " << filename << std::endl;
            return false;
        }
    }

    std::cout << "Successfully saved " << patterns_.size() << " pattern images to: " << savePath << std::endl;
    return true;
}
// 获取参数
void GrayCodeGenerator::getShadowMaskImages(cv::Mat& black, cv::Mat& white) const {
    black = blackImage_.clone();
    white = whiteImage_.clone();
}
// 运行
void GrayCodeGenerator::generateGrayCodePatterns() {
    std::cout << "\n=== Gray Code Pattern Generation ===\n";
    std::cout << "This will generate gray code patterns for projection.\n";

    GrayCodeGenerator::Params params;
    params.width = 860;
    params.height = 573;
    params.outputPath = "graycode";

    GrayCodeGenerator generator(params);

    std::cout << "Generating patterns (" << params.width << "x" << params.height << ")...\n";
    if (!generator.generatePatterns()) {
        std::cerr << "Failed to generate gray code patterns!\n";
        return;
    }

    std::cout << "Saving patterns to '" << params.outputPath << "' folder...\n";
    if (!generator.savePatterns()) {
        std::cerr << "Failed to save patterns!\n";
        return;
    }

    std::cout << "\nSuccessfully generated " << generator.getPatternCount()
        << " gray code patterns.\n";
    std::cout << "These patterns are ready for projection.\n";
}
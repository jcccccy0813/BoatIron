#include "GrayCodeDecoder.h"


GrayCodeDecoder::GrayCodeDecoder()
    : cameraSide_(LEFT_CAMERA) {
    setupCameraSpecificParams();
    initializeGrayCode();
}

GrayCodeDecoder::GrayCodeDecoder(CameraSide side)
    : cameraSide_(side) {
    setupCameraSpecificParams();
    initializeGrayCode();
}

GrayCodeDecoder::GrayCodeDecoder(const Params& params, CameraSide side)
    : params_(params), cameraSide_(side) {
    setupCameraSpecificParams();
    initializeGrayCode();
}

void GrayCodeDecoder::setCameraSide(CameraSide side) {
    cameraSide_ = side;
    setupCameraSpecificParams();
    initializeGrayCode();
}

void GrayCodeDecoder::setParams(const Params& params) {
    params_ = params;
    setupCameraSpecificParams();
    initializeGrayCode();
}

void GrayCodeDecoder::setupCameraSpecificParams() {
    // 硬编码的参数设置
    params_.width = 860;
    params_.height = 573;
    params_.whiteThreshold = 5;
    params_.blackThreshold = 45;

    // 根据相机侧别设置输出文件名
    if (cameraSide_ == LEFT_CAMERA) {
        params_.maskOutput = "mask_left.png";
        params_.xPngOutput = "x_left.png";
        params_.yPngOutput = "y_left.png";
        params_.xExrOutput = "x_left.exr";
        params_.yExrOutput = "y_left.exr";
    }
    else {
        params_.maskOutput = "mask_right.png";
        params_.xPngOutput = "x_right.png";
        params_.yPngOutput = "y_right.png";
        params_.xExrOutput = "x_right.exr";
        params_.yExrOutput = "y_right.exr";
    }
}

bool GrayCodeDecoder::initializeGrayCode() {
    cv::structured_light::GrayCodePattern::Params gcParams;
    gcParams.width = params_.width;
    gcParams.height = params_.height;

    graycode_ = cv::structured_light::GrayCodePattern::create(gcParams);
    if (!graycode_.empty()) {
        graycode_->setWhiteThreshold(params_.whiteThreshold);
    }
    return !graycode_.empty();
}

std::vector<std::string> GrayCodeDecoder::loadImageList(const std::string& filename) {
    std::vector<std::string> strList;
    std::ifstream ifs(filename.c_str());
    std::string tmp;
    while (ifs && getline(ifs, tmp)) {
        if (!tmp.empty()) {
            strList.push_back(tmp);
        }
    }
    return strList;
}

std::vector<cv::Mat> GrayCodeDecoder::loadImages(const std::vector<std::string>& filenames) {
    std::vector<cv::Mat> imgs;
    for (const auto& filename : filenames) {
        cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Warning: Failed to load image: " << filename << std::endl;
            continue;
        }
        imgs.push_back(img);
    }
    return imgs;
}

cv::Mat GrayCodeDecoder::computeShadowMask(const cv::Mat& blackImage, const cv::Mat& whiteImage) {
    cv::Mat shadowMask = cv::Mat::zeros(blackImage.size(), CV_8UC1);
    for (int j = 0; j < shadowMask.rows; ++j) {
        for (int i = 0; i < shadowMask.cols; ++i) {
            if (whiteImage.at<uchar>(j, i) > blackImage.at<uchar>(j, i) + params_.blackThreshold) {
                shadowMask.at<uchar>(j, i) = 255;
            }
        }
    }
    return shadowMask;
}

cv::Mat GrayCodeDecoder::computeDecodeImage(const std::vector<cv::Mat>& capturedPattern, const cv::Mat& mask) {
    cv::Mat decodedImage = cv::Mat::zeros(mask.size(), CV_32FC2);
    size_t numPatterns = graycode_->getNumberOfPatternImages();

    for (int j = 0; j < decodedImage.rows; ++j) {
        for (int i = 0; i < decodedImage.cols; ++i) {
            if (mask.at<uchar>(j, i) == 0) {
                continue;
            }

            cv::Point projPixel;
            bool error = graycode_->getProjPixel(capturedPattern, i, j, projPixel);
            if (!error) {
                decodedImage.at<cv::Vec2f>(j, i)[0] = static_cast<float>(projPixel.x);
                decodedImage.at<cv::Vec2f>(j, i)[1] = static_cast<float>(projPixel.y);
            }
        }
    }
    return decodedImage;
}

bool GrayCodeDecoder::decode() {
    // 根据相机侧别自动选择图像列表文件
    std::string imageListFile = (cameraSide_ == LEFT_CAMERA) ?
        "left_pattern_images.txt" : "right_pattern_images.txt";
    return decode(imageListFile);
}

bool GrayCodeDecoder::decode(const std::string& imageListFile) {
    std::vector<std::string> imageFiles = loadImageList(imageListFile);
    if (imageFiles.empty()) {
        std::cerr << "No images found in the list file: " << imageListFile << std::endl;
        return false;
    }
    return decode(imageFiles);
}

bool GrayCodeDecoder::decode(const std::vector<std::string>& imageFiles) {
    if (imageFiles.size() < 2) {
        std::cerr << "Need at least 2 images for decoding!" << std::endl;
        return false;
    }

    if (graycode_.empty()) {
        std::cerr << "GrayCode pattern decoder not initialized!" << std::endl;
        return false;
    }

    size_t numPatterns = graycode_->getNumberOfPatternImages();
    if (imageFiles.size() < numPatterns + 2) {
        std::cerr << "Insufficient images. Expected " << numPatterns + 2
            << " images, got " << imageFiles.size() << std::endl;
        return false;
    }

    std::cout << "Loading " << imageFiles.size() << " pattern images for "
        << (cameraSide_ == LEFT_CAMERA ? "left" : "right") << " camera..." << std::endl;

    std::vector<cv::Mat> capturedPattern = loadImages(imageFiles);
    if (capturedPattern.size() != imageFiles.size()) {
        std::cerr << "Failed to load some images!" << std::endl;
        return false;
    }

    // 获取黑白图像
    cv::Mat whiteImage = capturedPattern[capturedPattern.size() - 2];
    cv::Mat blackImage = capturedPattern[capturedPattern.size() - 1];

    std::cout << "Computing shadow mask..." << std::endl;
    shadowMask_ = computeShadowMask(blackImage, whiteImage);

    std::cout << "Decoding pattern..." << std::endl;
    decodedImage_ = computeDecodeImage(capturedPattern, shadowMask_);

    std::cout << "Generating visualization mask..." << std::endl;
    visualizationMask_ = getDecodedMask(decodedImage_);

    std::cout << "Decoding completed successfully for "
        << (cameraSide_ == LEFT_CAMERA ? "left" : "right") << " camera!" << std::endl;
    return true;
}

cv::Mat GrayCodeDecoder::getDecodedMask(const cv::Mat& decoded) {
    cv::Mat mask = cv::Mat::zeros(decoded.size(), CV_8UC1);
    for (int j = 0; j < decoded.rows; ++j) {
        for (int i = 0; i < decoded.cols; ++i) {
            if (decoded.at<cv::Vec2f>(j, i)[0] != 0.0f) {
                mask.at<uchar>(j, i) = 255;
            }
        }
    }
    return mask;
}

void GrayCodeDecoder::visualizeDecodedImage(const cv::Mat& decoded,
    const std::string& xPath,
    const std::string& yPath) {
    cv::Mat xMap = cv::Mat::zeros(decoded.size(), CV_8UC1);
    cv::Mat yMap = cv::Mat::zeros(decoded.size(), CV_8UC1);

    for (int j = 0; j < decoded.rows; ++j) {
        for (int i = 0; i < decoded.cols; ++i) {
            if (decoded.at<cv::Vec2f>(j, i)[0] == 0.0f) {
                continue;
            }
            int corresX = static_cast<int>(decoded.at<cv::Vec2f>(j, i)[0] * 255.0f / params_.width);
            int corresY = static_cast<int>(decoded.at<cv::Vec2f>(j, i)[1] * 255.0f / params_.height);
            xMap.at<uchar>(j, i) = cv::saturate_cast<uchar>(corresX);
            yMap.at<uchar>(j, i) = cv::saturate_cast<uchar>(corresY);
        }
    }

    cv::imwrite(xPath, xMap);
    cv::imwrite(yPath, yMap);
}

void GrayCodeDecoder::saveDecodedImage(const cv::Mat& decoded,
    const std::string& xPath,
    const std::string& yPath) {
    std::vector<cv::Mat> channels;
    cv::split(decoded, channels);
    cv::imwrite(xPath, channels[0]);
    cv::imwrite(yPath, channels[1]);
}

bool GrayCodeDecoder::saveResults() {
    if (decodedImage_.empty()) {
        std::cerr << "No decoded data to save!" << std::endl;
        return false;
    }

    std::cout << "Saving results for "
        << (cameraSide_ == LEFT_CAMERA ? "left" : "right") << " camera..." << std::endl;

    // 保存可视化结果
    visualizeDecodedImage(decodedImage_, params_.xPngOutput, params_.yPngOutput);

    // 保存浮点精度结果
    saveDecodedImage(decodedImage_, params_.xExrOutput, params_.yExrOutput);

    // 保存掩码
    cv::imwrite(params_.maskOutput, visualizationMask_);

    std::cout << "Results saved:" << std::endl;
    std::cout << "- Mask: " << params_.maskOutput << std::endl;
    std::cout << "- X visualization: " << params_.xPngOutput << std::endl;
    std::cout << "- Y visualization: " << params_.yPngOutput << std::endl;
    std::cout << "- X data (float): " << params_.xExrOutput << std::endl;
    std::cout << "- Y data (float): " << params_.yExrOutput << std::endl;

    return true;
}

void GrayCodeDecoder::decodeGrayCodePatterns() {
    std::cout << "\n=== Gray Code Decoding ===\n";
    std::cout << "This will decode captured gray code patterns for:\n";
    std::cout << "1. Left camera only\n";
    std::cout << "2. Right camera only\n";
    std::cout << "3. Both cameras\n";
    std::cout << "Enter choice (1-3): ";

    int choice;
    if (!(std::cin >> choice) || choice < 1 || choice > 3) {
        clearInputBuffer();
        std::cerr << "Invalid choice.\n";
        return;
    }
    clearInputBuffer();

    GrayCodeDecoder::Params params;
    params.width = 860;
    params.height = 573;
    params.whiteThreshold = 5;
    params.blackThreshold = 45;

    bool success = true;

    if (choice == 1 || choice == 3) {
        std::cout << "\nDecoding left camera patterns...\n";
        GrayCodeDecoder leftDecoder(params, GrayCodeDecoder::LEFT_CAMERA);
        if (!leftDecoder.decode("left_pattern_images.txt")) {
            std::cerr << "Failed to decode left camera patterns!\n";
            success = false;
        }
        else {
            leftDecoder.saveResults();
            std::cout << "Left camera decoding completed.\n";
        }
    }

    if (choice == 2 || choice == 3) {
        std::cout << "\nDecoding right camera patterns...\n";
        GrayCodeDecoder rightDecoder(params, GrayCodeDecoder::RIGHT_CAMERA);
        if (!rightDecoder.decode("right_pattern_images.txt")) {
            std::cerr << "Failed to decode right camera patterns!\n";
            success = false;
        }
        else {
            rightDecoder.saveResults();
            std::cout << "Right camera decoding completed.\n";
        }
    }

    if (success) {
        std::cout << "\nGray code decoding completed successfully!\n";
        std::cout << "Output files saved in current directory.\n";
    }
    else {
        std::cerr << "\nGray code decoding completed with errors.\n";
    }
}

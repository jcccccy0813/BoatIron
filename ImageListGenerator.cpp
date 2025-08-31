#include "ImageListGenerator.h"

// 辅助函数
int ImageListGenerator::extractNumberFromFilename(const std::string& filename) const {
    std::smatch match;
    std::regex pattern(R"((\d+))");
    if (std::regex_search(filename, match, pattern)) {
        return std::stoi(match[1]);
    }
    return -1;
}
// 获取所有图像文件
std::vector<fs::directory_entry> ImageListGenerator::getImageFiles(const std::string& folderPath) const {
    std::vector<fs::directory_entry> imageFiles;

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                imageFiles.push_back(entry);
            }
        }
    }

    return imageFiles;
}
// 生成图像列表
void ImageListGenerator::sortImageFiles(std::vector<fs::directory_entry>& files) const {
    std::sort(files.begin(), files.end(), [this](const fs::directory_entry& a, const fs::directory_entry& b) {
        std::string nameA = a.path().filename().string();
        std::string nameB = b.path().filename().string();

        // 特殊处理参考图像
        if (nameA == "black_ref.png") return false;
        if (nameB == "black_ref.png") return true;
        if (nameA == "white_ref.png") return false;
        if (nameB == "white_ref.png") return true;

        return extractNumberFromFilename(nameA) < extractNumberFromFilename(nameB);
        });
}
// 写入图像列表文件
bool ImageListGenerator::writeToOutputFile(const std::vector<fs::directory_entry>& files,
    const std::string& folderPath,
    const std::string& outputFilePath) const {
    std::ofstream outFile(outputFilePath);
    if (!outFile.is_open()) {
        std::cerr << "无法打开输出文件: " << outputFilePath << std::endl;
        return false;
    }

    for (const auto& file : files) {
        outFile << folderPath << "/" << file.path().filename().string() << std::endl;
    }

    outFile.close();
    return true;
}
// 主函数
bool ImageListGenerator::generateImageList(const std::string& folderPath, const std::string& outputFilePath) {
    // 检查输入文件夹是否存在
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        std::cerr << "无效的文件夹路径: " << folderPath << std::endl;
        return false;
    }

    // 获取所有图像文件
    auto imageFiles = getImageFiles(folderPath);
    if (imageFiles.empty()) {
        std::cerr << "文件夹中没有找到图像文件: " << folderPath << std::endl;
        return false;
    }

    // 对文件进行排序
    sortImageFiles(imageFiles);

    // 写入输出文件
    if (!writeToOutputFile(imageFiles, folderPath, outputFilePath)) {
        return false;
    }

    std::cout << "成功生成图像列表文件: " << outputFilePath
        << " (包含 " << imageFiles.size() << " 个图像)" << std::endl;
    return true;
}
//运行
void ImageListGenerator::generateImageList() {
    std::cout << "\n=== Image List Generation ===\n";
    std::cout << "This will generate a sorted list of image files.\n";

    std::string folderPath, outputFile;

    std::cout << "Enter image folder path: ";
    std::cin >> folderPath;
    clearInputBuffer();

    std::cout << "Enter output text file path: ";
    std::cin >> outputFile;
    clearInputBuffer();

    ImageListGenerator generator;
    if (generator.generateImageList(folderPath, outputFile)) {
        std::cout << "Image list generated successfully!\n";
    }
    else {
        std::cerr << "Failed to generate image list.\n";
    }
}

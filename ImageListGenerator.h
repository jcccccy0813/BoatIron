#ifndef IMAGE_LIST_GENERATOR_H
#define IMAGE_LIST_GENERATOR_H

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <algorithm>
#include"Common.h"

namespace fs = std::filesystem;

class ImageListGenerator {
public:
    ImageListGenerator() = default;

    /**
     * @brief 从指定文件夹生成图像文件列表并保存到输出文件
     * @param folderPath 包含图像文件的文件夹路径
     * @param outputFilePath 输出文本文件路径
     * @return 成功返回true，失败返回false
     */
    bool generateImageList(const std::string& folderPath, const std::string& outputFilePath);
    static void generateImageList();

private:
    /**
     * @brief 从文件名中提取数字
     * @param filename 文件名
     * @return 提取的数字，失败返回-1
     */
    int extractNumberFromFilename(const std::string& filename) const;

    /**
     * @brief 获取文件夹中的所有图像文件
     * @param folderPath 文件夹路径
     * @return 图像文件路径向量
     */
    std::vector<fs::directory_entry> getImageFiles(const std::string& folderPath) const;

    /**
     * @brief 对图像文件进行排序
     * @param files 要排序的文件向量
     */
    void sortImageFiles(std::vector<fs::directory_entry>& files) const;

    /**
     * @brief 将文件列表写入输出文件
     * @param files 文件向量
     * @param folderPath 原始文件夹路径
     * @param outputFilePath 输出文件路径
     * @return 成功返回true，失败返回false
     */
    bool writeToOutputFile(const std::vector<fs::directory_entry>& files,
        const std::string& folderPath,
        const std::string& outputFilePath) const;
};

#endif // IMAGE_LIST_GENERATOR_H
#ifndef CAMERA_SYNC_CAPTURE_H
#define CAMERA_SYNC_CAPTURE_H

#include <windows.h>
#include <gdiplus.h>
#include <filesystem>
#include <vector>
#include <string>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include "CameraManager.h"  // 替换 CameraCapture.h

class CameraSyncCapture {
public:
    CameraSyncCapture();
    ~CameraSyncCapture();

    // 初始化
    bool initialize();

    // 主运行函数
    void runSyncCapture();

    // 停止采集
    void stopCapture();

    // 状态查询
    bool isRunning() const { return isRunning_; }  // 修改：使用自己的运行状态

    // 加载图像文件
    static std::vector<std::wstring> loadImageFiles(const std::wstring& folder);
    static void structuredLightCapture();

private:
    // 投影相关
    bool initializeProjector();
    void projectImage(const std::wstring& imagePath);

    // 图像保存处理
    void triggerImageSave();  // 修改：触发图像保存

    // 同步等待
    void waitForCamerasToSave();

    // 成员变量
    CameraManager cameraManager_;  // 替换 CameraCapture cameraCapture_;
    HWND hwnd_ = nullptr;
    HDC hdc_ = nullptr;
    ULONG_PTR gdiplusToken_ = 0;

    std::atomic<int> currentGroup_{ 0 };
    std::atomic<int> imagesSaved_{ 0 };
    std::atomic<bool> isRunning_{ false };  // 添加：自己的运行状态
    std::mutex syncMutex_;
    std::condition_variable syncCV_;
};

#endif // CAMERA_SYNC_CAPTURE_H
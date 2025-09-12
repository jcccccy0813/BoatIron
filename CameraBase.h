#ifndef CAMERA_BASE_H
#define CAMERA_BASE_H

#include <string>
#include <atomic>
#include <mutex>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "MvCameraControl.h"

#ifdef _WIN32
#include <direct.h>
#include <errno.h>
#else
#include <sys/types.h>
#endif
class CameraBase {
public:
    /**
     * @brief 相机配置参数结构体
     */
    struct CameraConfig {
        std::string windowName;
        std::string cameraName;
        int width = 1920;
        int height = 1080;
        float gamma = 0.37f;
        float exposureTime = 10000.0f;
        bool triggerMode = false; // 0: Off, 1: On
    };
    // 构造函数
    CameraBase();
    virtual ~CameraBase();

    // 初始化相关
    bool initialize(int index = 0);
    // 枚举相机
    static bool enumDevices(MV_CC_DEVICE_INFO_LIST& deviceList);

    // 相机控制
    // 分辨率设置
    bool setResolution(int width, int height);
    // Gamma设置
    bool startGrabbing();
    // 停止采集
    bool stopGrabbing();
    // 关闭相机
    bool closeDevice();

    // 参数设置
    // Gamma设置
    bool setGamma(float gamma);
    // 曝光时间设置
    bool setExposureTime(float exposureTime);
    // 触发模式设置
    bool setTriggerMode(bool enabled);

    // 图像采集
    // 获取一帧图像
    cv::Mat captureFrame(int timeout = 1000);

    // 工具函数
    // 创建目录
    static bool createDirectoryIfNotExists(const std::string& dir);
    // 保存图像
    bool saveFrame(const cv::Mat& frame, const std::string& filename, int quality = 90);

    // 获取信息
    // 获取句柄
    void* getHandle() const { return handle_; }
    // 获取配置
    const CameraConfig& getConfig() const { return config_; }
    // 获取名称
    bool isRunning() const { return isRunning_; }
    // 获取数据大小
    unsigned int getPayloadSize() const { return payloadSize_; }

    // 配置设置
    // 设置配置
    void setConfig(const CameraConfig& config) { config_ = config; }
    // 设置窗口名称
    void setWindowName(const std::string& name) { config_.windowName = name; }
    // 设置相机名称
    void setCameraName(const std::string& name) { config_.cameraName = name; }

protected:
    void* handle_ = nullptr;
    CameraConfig config_;
    unsigned int payloadSize_ = 0;
    std::atomic<bool> isRunning_{ false };

private:
    bool setupCameraParameters();
};

#endif // CAMERA_BASE_H
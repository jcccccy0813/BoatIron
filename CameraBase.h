#ifndef CAMERA_BASE_H
#define CAMERA_BASE_H

#include <string>
#include <atomic>
#include <mutex>
#include <vector>
#include <opencv2/core.hpp>
#include "MvCameraControl.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#include <direct.h>
#include <errno.h>
#else
#include <sys/types.h>
#endif
class CameraBase {
public:
    struct CameraConfig {
        std::string windowName;
        std::string cameraName;
        int width = 1920;
        int height = 1080;
        float gamma = 0.37f;
        float exposureTime = 10000.0f;
        bool triggerMode = false; // 0: Off, 1: On
    };

    CameraBase();
    virtual ~CameraBase();

    // 初始化相关
    bool initialize(int index = 0);
    static bool enumDevices(MV_CC_DEVICE_INFO_LIST& deviceList);

    // 相机控制
    bool setResolution(int width, int height);
    bool startGrabbing();
    bool stopGrabbing();
    bool closeDevice();

    // 参数设置
    bool setGamma(float gamma);
    bool setExposureTime(float exposureTime);
    bool setTriggerMode(bool enabled);

    // 图像采集
    cv::Mat captureFrame(int timeout = 1000);

    // 工具函数
    static bool createDirectoryIfNotExists(const std::string& dir);
    bool saveFrame(const cv::Mat& frame, const std::string& filename, int quality = 90);

    // 获取信息
    void* getHandle() const { return handle_; }
    const CameraConfig& getConfig() const { return config_; }
    bool isRunning() const { return isRunning_; }
    unsigned int getPayloadSize() const { return payloadSize_; }

    // 配置设置
    void setConfig(const CameraConfig& config) { config_ = config; }
    void setWindowName(const std::string& name) { config_.windowName = name; }
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
#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H
#include"Common.h"
#include "CameraBase.h"
#include <thread>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <functional>
#include <iostream>
#include <iomanip>
#include <sstream>

class CameraCapture {
public:
    // 回调函数类型
    using ImageSavedCallback = std::function<void(const std::string&, const std::string&)>;

    CameraCapture();
    ~CameraCapture();

    // 初始化函数
    bool initSingleCamera(int index);
    bool initDualCameras();

    // 采集控制
    void startCapture();
    void stopCapture();

    // 图像保存
    void saveImages(int groupId, bool isSingleMode = false);

    // 获取图像列表
    const std::vector<std::string>& getLeftImages() const;
    const std::vector<std::string>& getRightImages() const;

    // 状态查询
    bool isRunning() const { return globalRunning_; }
    bool isSingleMode() const { return !rightCamera_.isRunning(); }

    // 设置回调函数
    void setImageSavedCallback(ImageSavedCallback callback) {
        imageSavedCallback_ = callback;
    }
    static void captureSingleCamera (); 
    static void captureDualCameras();
private:
    // 相机线程函数
    void cameraThread(CameraBase* camera, bool isSingleMode);

    // 内部保存函数
    void saveFrameWithCallback(CameraBase* camera, const cv::Mat& frame,
        const std::string& folder, int groupId);

    // 相机实例
    CameraBase leftCamera_;
    CameraBase rightCamera_;
    std::thread leftThread_;
    std::thread rightThread_;

    // 同步控制
    std::atomic<bool> globalRunning_{ true };
    std::atomic<bool> globalSave_{ false };
    std::atomic<int> saveCount_{ 0 };
    std::atomic<int> saveGroupID_{ 0 };
    std::mutex saveMutex_;

    // 图像存储
    std::vector<std::string> leftImages_;
    std::vector<std::string> rightImages_;

    // 回调函数
    ImageSavedCallback imageSavedCallback_;
};

#endif // CAMERA_CAPTURE_H
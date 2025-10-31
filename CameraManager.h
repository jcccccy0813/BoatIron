#ifndef CAMERA_MANAGER_H
#define CAMERA_MANAGER_H

#include "CameraBase.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>

class CameraManager {
public:
    CameraManager();
    ~CameraManager();

    // 单目/双目相机模式
    void runSingleCameraMode();
    void runDualCameraMode();
    void CameraManager::triggerCapture();
    bool CameraManager::isSaving() const { return globalSave_.load(); }
    int CameraManager::getSaveCount() const { return saveCount_.load(); }
    static constexpr const char* DATA_FOLDER = "data";
    static constexpr const char* LEFT_CAM_FOLDER = "Cam_001";
    static constexpr const char* RIGHT_CAM_FOLDER = "Cam_002";
private:
    struct CameraThread {
        CameraBase camera;
        std::thread thread;
        std::string name;
        std::atomic<bool> running{ false };
        std::atomic<bool> readyToStart{ false };
    };

    // 相机循环
    void cameraLoop(CameraThread* cam, bool isSingle = false);

    // 工具函数
    int detectLeftRightCameras(const MV_CC_DEVICE_INFO_LIST& deviceList, int& leftIndex, int& rightIndex);
    void handleKeyboardInput(int cameraCount);
    //置1
    void resetSaveGroupID() { saveGroupID_ = 0; }
    std::atomic<bool> globalRunning_{ true };
    std::atomic<bool> globalSave_{ false };
    std::atomic<int> saveCount_{ 0 };
    std::atomic<int> saveGroupID_{ 0 };
    std::mutex saveMutex_;
};

#endif // CAMERA_MANAGER_H
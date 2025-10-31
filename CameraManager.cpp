#include "CameraManager.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

using namespace std;

CameraManager::CameraManager() {}
CameraManager::~CameraManager() {
    globalRunning_ = false;
}

// 检测左右相机
int CameraManager::detectLeftRightCameras(const MV_CC_DEVICE_INFO_LIST& deviceList, int& leftIndex, int& rightIndex) {
    vector<string> cameraNames;
    for (unsigned int i = 0; i < deviceList.nDeviceNum; ++i) {
        if (deviceList.pDeviceInfo[i]->nTLayerType == MV_GIGE_DEVICE)
            cameraNames.emplace_back((char*)deviceList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.chUserDefinedName);
        else
            cameraNames.emplace_back((char*)deviceList.pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chUserDefinedName);
    }

    leftIndex = rightIndex = -1;
    for (size_t i = 0; i < cameraNames.size(); ++i) {
        if (cameraNames[i].find("left") != string::npos || cameraNames[i].find("Left") != string::npos)
            leftIndex = i;
        else if (cameraNames[i].find("right") != string::npos || cameraNames[i].find("Right") != string::npos)
            rightIndex = i;
    }
    if (leftIndex == -1 && rightIndex == -1 && deviceList.nDeviceNum >= 2) {
        leftIndex = 0;
        rightIndex = 1;
    }

    return (leftIndex != -1 && rightIndex != -1) ? 2 : 1;
}

// 相机循环
void CameraManager::cameraLoop(CameraThread* cam, bool isSingle) {
    while (!cam->readyToStart && globalRunning_)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    cam->running = true;
    cam->camera.startGrabbing();

    cv::namedWindow(cam->name, cv::WINDOW_AUTOSIZE);
    while (globalRunning_ && cam->running) {
        cv::Mat frame = cam->camera.captureFrame();
        if (!frame.empty()) {
            cv::imshow(cam->name, frame);
            cv::waitKey(1);

            if (globalSave_) {
                std::lock_guard<std::mutex> lock(saveMutex_);
                int group = saveGroupID_.load();

                // === 修改保存路径：从 stereo 改为 data ===
                std::string folder;
                std::string filename;

                if (isSingle) {
                    // 单目模式：data/leftsingle 或 data/rightsingle
                    folder = (cam->name == "left") ? "data/leftsingle" : "data/rightsingle";
                    CameraBase::createDirectoryIfNotExists(folder);

                    std::ostringstream oss;
                    oss << folder << "/" << cam->name << std::setw(2) << std::setfill('0') << group << ".jpg";
                    filename = oss.str();

                    // 单目模式使用JPEG格式
                    if (cam->camera.saveFrame(frame, filename, 90))
                        printf("[%s] Saved: %s\n", cam->name.c_str(), filename.c_str());
                    else
                        printf("[%s] Save failed!\n", cam->name.c_str());
                }
                else {
                    // 双目模式：data/Cam_001 和 data/Cam_002
                    std::string camFolder = (cam->name == "left") ? LEFT_CAM_FOLDER : RIGHT_CAM_FOLDER;
                    folder = std::string(DATA_FOLDER) + "/" + camFolder;
                    CameraBase::createDirectoryIfNotExists(folder);

                    std::ostringstream oss;
                    oss << folder << "/img_" << std::setw(4) << std::setfill('0') << group << ".png";
                    filename = oss.str();

                    // 双目模式使用PNG格式，不压缩
                    std::vector<int> params = { cv::IMWRITE_PNG_COMPRESSION, 0 };
                    if (cv::imwrite(filename, frame, params))
                        printf("[%s] Saved: %s\n", cam->name.c_str(), filename.c_str());
                    else
                        printf("[%s] Save failed!\n", cam->name.c_str());
                }

                if (--saveCount_ == 0) {
                    globalSave_ = false;
                    printf("All cameras saved for group %d.\n", group);
                }
            }
        }
    }

    cam->camera.stopGrabbing();
    cv::destroyWindow(cam->name);
}

// 键盘输入处理
void CameraManager::handleKeyboardInput(int cameraCount) {
    printf("Press 'S' to save, 'Q' to quit.\n");
    while (globalRunning_) {
        int key = getchar();
        if (key == 's' || key == 'S') {
            globalSave_ = true;
            saveCount_ = cameraCount;
            ++saveGroupID_;
        }
        else if (key == 'q' || key == 'Q') {
            globalRunning_ = false;
        }
    }
}

// 单相机模式
void CameraManager::runSingleCameraMode() {
    globalRunning_ = true;
    globalSave_ = false;
    saveCount_ = 0;
    resetSaveGroupID();
    MV_CC_DEVICE_INFO_LIST deviceList = { 0 };
    if (!CameraBase::enumDevices(deviceList) || deviceList.nDeviceNum == 0) {
        printf("No cameras found.\n");
        return;
    }

    int index;
    printf("Enter camera index (1 for left, 2 for right): ");
    cin >> index;
    index -= 1;

    // 检测左右相机
    int leftIndex, rightIndex;
    detectLeftRightCameras(deviceList, leftIndex, rightIndex);

    // 确定实际要打开的相机索引
    int actualIndex = index;
    if (index == 0 && leftIndex != -1) {
        actualIndex = leftIndex;
    }
    else if (index == 1 && rightIndex != -1) {
        actualIndex = rightIndex;
    }

    CameraThread cam;
    cam.name = (index == 0) ? "left" : "right";
    cam.readyToStart = true;

    if (!cam.camera.initialize(actualIndex)) {
        printf("Failed to initialize camera %d\n", actualIndex);
        return;
    }

    // 设置相机参数
    cam.camera.setTriggerMode(false);
    cam.camera.setGamma(0.37f);

    printf("Opening %s camera (device index %d)\n", cam.name.c_str(), actualIndex);

    std::thread t(&CameraManager::cameraLoop, this, &cam, true);
    handleKeyboardInput(1);
    cam.running = false;

    if (t.joinable()) t.join();
}

// 双相机模式
void CameraManager::runDualCameraMode() {
    globalRunning_ = true;
    globalSave_ = false;
    saveCount_ = 0; 
    resetSaveGroupID();
    MV_CC_DEVICE_INFO_LIST deviceList = { 0 };
    if (!CameraBase::enumDevices(deviceList) || deviceList.nDeviceNum < 2) {
        printf("Need at least 2 cameras!\n");
        return;
    }

    int leftIndex, rightIndex;
    detectLeftRightCameras(deviceList, leftIndex, rightIndex);
    if (leftIndex == -1 || rightIndex == -1) {
        printf("Failed to identify left/right cameras.\n");
        return;
    }

    CameraThread leftCam, rightCam;
    leftCam.name = "left";
    rightCam.name = "right";
    leftCam.readyToStart = true;
    rightCam.readyToStart = true;

    // 获取相机名称用于显示
    vector<string> cameraNames;
    for (unsigned int i = 0; i < deviceList.nDeviceNum; ++i) {
        if (deviceList.pDeviceInfo[i]->nTLayerType == MV_GIGE_DEVICE)
            cameraNames.emplace_back((char*)deviceList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.chUserDefinedName);
        else
            cameraNames.emplace_back((char*)deviceList.pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chUserDefinedName);
    }

    if (!leftCam.camera.initialize(leftIndex) || !rightCam.camera.initialize(rightIndex)) {
        printf("Failed to initialize both cameras.\n");
        return;
    }

    // 设置相机参数
    leftCam.camera.setTriggerMode(false);
    leftCam.camera.setGamma(0.37f);
    rightCam.camera.setTriggerMode(false);
    rightCam.camera.setGamma(0.37f);

    printf("Opening cameras:\n");
    printf("- Left camera: index %d, name '%s'\n", leftIndex, cameraNames[leftIndex].c_str());
    printf("- Right camera: index %d, name '%s'\n", rightIndex, cameraNames[rightIndex].c_str());

    std::thread tLeft(&CameraManager::cameraLoop, this, &leftCam, false);
    std::thread tRight(&CameraManager::cameraLoop, this, &rightCam, false);
    handleKeyboardInput(2);

    leftCam.running = rightCam.running = false;
    if (tLeft.joinable()) tLeft.join();
    if (tRight.joinable()) tRight.join();
}
void CameraManager::triggerCapture() {
    globalSave_ = true;
    saveCount_ = 2; // 双相机模式
    ++saveGroupID_;
    std::cout << "Programmatic capture triggered, group: " << saveGroupID_.load() << std::endl;
}


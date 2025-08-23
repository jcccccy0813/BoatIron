#include "CameraCapture.h"


CameraCapture::CameraCapture(): imageSavedCallback_(nullptr) {
    // 配置左相机
    CameraBase::CameraConfig leftConfig;
    leftConfig.windowName = "Left Camera";
    leftConfig.cameraName = "left";
    leftCamera_.setConfig(leftConfig);

    // 配置右相机
    CameraBase::CameraConfig rightConfig;
    rightConfig.windowName = "Right Camera";
    rightConfig.cameraName = "right";
    rightCamera_.setConfig(rightConfig);
}

CameraCapture::~CameraCapture() {
    stopCapture();
}

bool CameraCapture::initSingleCamera(int index) {
    if (index < 0 || index > 1) return false;

    if (index == 0) {
        return leftCamera_.initialize(0);
    }
    else {
        return rightCamera_.initialize(0); // 对于单相机模式，右相机也使用第一个设备
    }
}

bool CameraCapture::initDualCameras() {
    return leftCamera_.initialize(0) && rightCamera_.initialize(1);
}

void CameraCapture::startCapture() {
    globalRunning_ = true;

    if (leftCamera_.getHandle()) {
        leftCamera_.startGrabbing();
        leftThread_ = std::thread(&CameraCapture::cameraThread, this, &leftCamera_, isSingleMode());
    }

    if (rightCamera_.getHandle()) {
        rightCamera_.startGrabbing();
        rightThread_ = std::thread(&CameraCapture::cameraThread, this, &rightCamera_, isSingleMode());
    }
}

void CameraCapture::saveFrameWithCallback(CameraBase* camera, const cv::Mat& frame,
    const std::string& folder, int groupId) {
    CameraBase::createDirectoryIfNotExists(folder);

    std::ostringstream oss;
    oss << folder << "/" << camera->getConfig().cameraName
        << std::setw(2) << std::setfill('0') << groupId << ".jpg";
    std::string filename = oss.str();

    if (camera->saveFrame(frame, filename, 90)) {
        std::cout << "[" << camera->getConfig().cameraName << "] Saved: " << filename << std::endl;

        // 添加到图像列表
        if (camera->getConfig().cameraName == "left") {
            leftImages_.push_back(filename);
        }
        else {
            rightImages_.push_back(filename);
        }

        // 调用回调函数
        if (imageSavedCallback_) {
            imageSavedCallback_(camera->getConfig().cameraName, filename);
        }
    }
    else {
        std::cerr << "[" << camera->getConfig().cameraName << "] Save failed!" << std::endl;
    }
}

void CameraCapture::cameraThread(CameraBase* camera, bool isSingleMode) {
    cv::namedWindow(camera->getConfig().windowName, cv::WINDOW_AUTOSIZE);

    while (globalRunning_ && camera->isRunning()) {
        cv::Mat frame = camera->captureFrame();
        if (!frame.empty()) {
            cv::imshow(camera->getConfig().windowName, frame);
            cv::waitKey(1);

            // 保存图像逻辑
            if (globalSave_.load()) {
                std::lock_guard<std::mutex> lock(saveMutex_);
                int group = saveGroupID_.load();

                std::string folder;
                if (isSingleMode) {
                    folder = (camera->getConfig().cameraName == "left") ? "leftsingle" : "rightsingle";
                }
                else {
                    folder = "stereo";
                }

                saveFrameWithCallback(camera, frame, folder, group);

                if (--saveCount_ == 0) {
                    globalSave_ = false;
                    std::cout << "All cameras have saved images for group " << group << std::endl;
                }
            }
        }
    }

    cv::destroyWindow(camera->getConfig().windowName);
}

void CameraCapture::stopCapture() {
    globalRunning_ = false;

    if (leftThread_.joinable()) {
        leftThread_.join();
    }
    if (rightThread_.joinable()) {
        rightThread_.join();
    }

    leftCamera_.stopGrabbing();
    rightCamera_.stopGrabbing();
}

void CameraCapture::saveImages(int groupId, bool isSingleMode) {
    std::lock_guard<std::mutex> lock(saveMutex_);
    globalSave_ = true;
    saveCount_ = isSingleMode ? 1 : 2;
    saveGroupID_ = groupId;
}

const std::vector<std::string>& CameraCapture::getLeftImages() const {
    return leftImages_;
}

const std::vector<std::string>& CameraCapture::getRightImages() const {
    return rightImages_;
}

void CameraCapture::captureSingleCamera() {
    CameraCapture capture;
    int index;
    std::cout << "\nSelect camera to capture:\n";
    std::cout << "0. Left Camera\n";
    std::cout << "1. Right Camera\n";
    std::cout << "Enter choice (0-1): ";

    if (!(std::cin >> index) || (index != 0 && index != 1)) {
        clearInputBuffer();
        std::cerr << "Invalid camera selection.\n";
        return;
    }

    std::string camName = (index == 0) ? "Left" : "Right";
    std::cout << "\n=== " << camName << " Camera Capture ===\n";

    if (!capture.initSingleCamera(index)) {
        std::cerr << "Failed to initialize " << camName << " camera!\n";
        return;
    }

    capture.startCapture();
    std::cout << "Camera streaming started. Commands:\n";
    std::cout << "  S - Save current frame\n";
    std::cout << "  Q - Quit capture mode\n";

    char cmd;
    while (true) {
        std::cout << "> ";
        std::cin >> cmd;
        clearInputBuffer();

        if (cmd == 's' || cmd == 'S') {
            static int groupId = 0;
            capture.saveImages(groupId++, true);
            std::cout << camName << " frame saved to "
                << (index == 0 ? "left" : "right")
                << "single folder\n";
        }
        else if (cmd == 'q' || cmd == 'Q') {
            break;
        }
    }
    capture.stopCapture();
}
void  CameraCapture::captureDualCameras() {
    CameraCapture capture;
    std::cout << "\n=== Dual Camera Capture ===\n";

    if (!capture.initDualCameras()) {
        std::cerr << "Failed to initialize cameras!\n";
        return;
    }

    capture.startCapture();
    std::cout << "D dual camera streaming started. Commands:\n";
    std::cout << "  S - Save stereo pair\n";
    std::cout << "  Q - Quit capture mode\n";

    char cmd;
    while (true) {
        std::cout << "> ";
        std::cin >> cmd;
        clearInputBuffer();

        if (cmd == 's' || cmd == 'S') {
            static int groupId = 0;
            capture.saveImages(groupId++);
            std::cout << "Stereo pair " << groupId << " saved to stereo folder\n";
        }
        else if (cmd == 'q' || cmd == 'Q') {
            break;
        }
    }
    capture.stopCapture();
}
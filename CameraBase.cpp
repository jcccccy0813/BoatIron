#include "CameraBase.h"

//窗口名相机名
CameraBase::CameraBase() {
    config_.windowName = "Camera";
    config_.cameraName = "camera";
}
//析构函数
CameraBase::~CameraBase() {
    closeDevice();
}
//枚举设备
bool CameraBase::enumDevices(MV_CC_DEVICE_INFO_LIST& deviceList) {
    return MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &deviceList) == MV_OK;
}
//初始化
bool CameraBase::initialize(int index) {
    //枚举设备
    MV_CC_DEVICE_INFO_LIST deviceList = { 0 };
    if (!enumDevices(deviceList) || index >= (int)deviceList.nDeviceNum) {
        std::cerr << "Camera not found at index: " << index << std::endl;
        return false;
    }
    //创建句柄
    if (MV_CC_CreateHandle(&handle_, deviceList.pDeviceInfo[index]) != MV_OK) {
        std::cerr << "Failed to create camera handle." << std::endl;
        return false;
    }
    //创建句柄
    if (MV_CC_OpenDevice(handle_) != MV_OK) {
        std::cerr << "Failed to open camera." << std::endl;
        return false;
    }

    // 设置相机参数
    if (!setupCameraParameters()) {
        return false;
    }

    // 获取payload大小
    MVCC_INTVALUE stParam = { 0 };
    if (MV_CC_GetIntValue(handle_, "PayloadSize", &stParam) != MV_OK) {
        std::cerr << "Failed to get payload size." << std::endl;
        return false;
    }
    payloadSize_ = stParam.nCurValue;

    return setResolution(config_.width, config_.height);
}
//设置参数
bool CameraBase::setupCameraParameters() {
    if (MV_CC_SetEnumValue(handle_, "TriggerMode", config_.triggerMode ? 1 : 0) != MV_OK) {
        std::cerr << "Failed to set trigger mode." << std::endl;
        return false;
    }

    if (MV_CC_SetBoolValue(handle_, "GammaEnable", true) != MV_OK) {
        std::cerr << "Failed to enable gamma." << std::endl;
        return false;
    }

    if (MV_CC_SetFloatValue(handle_, "Gamma", config_.gamma) != MV_OK) {
        std::cerr << "Failed to set gamma." << std::endl;
        return false;
    }

    if (MV_CC_SetFloatValue(handle_, "ExposureTime", config_.exposureTime) != MV_OK) {
        std::cerr << "Failed to set exposure time." << std::endl;
        return false;
    }

    return true;
}

bool CameraBase::setResolution(int width, int height) {
    int ret1 = MV_CC_SetIntValue(handle_, "Width", width);
    int ret2 = MV_CC_SetIntValue(handle_, "Height", height);
    return ret1 == MV_OK && ret2 == MV_OK;
}

bool CameraBase::startGrabbing() {
    if (MV_CC_StartGrabbing(handle_) == MV_OK) {
        isRunning_ = true;
        return true;
    }
    return false;
}

bool CameraBase::stopGrabbing() {
    if (isRunning_) {
        isRunning_ = false;
        return MV_CC_StopGrabbing(handle_) == MV_OK;
    }
    return true;
}

bool CameraBase::closeDevice() {
    stopGrabbing();
    if (handle_) {
        MV_CC_CloseDevice(handle_);
        MV_CC_DestroyHandle(handle_);
        handle_ = nullptr;
        return true;
    }
    return false;
}

bool CameraBase::setGamma(float gamma) {
    config_.gamma = gamma;
    if (handle_) {
        return MV_CC_SetFloatValue(handle_, "Gamma", gamma) == MV_OK;
    }
    return true;
}

bool CameraBase::setExposureTime(float exposureTime) {
    config_.exposureTime = exposureTime;
    if (handle_) {
        return MV_CC_SetFloatValue(handle_, "ExposureTime", exposureTime) == MV_OK;
    }
    return true;
}

bool CameraBase::setTriggerMode(bool enabled) {
    config_.triggerMode = enabled;
    if (handle_) {
        return MV_CC_SetEnumValue(handle_, "TriggerMode", enabled ? 1 : 0) == MV_OK;
    }
    return true;
}

cv::Mat CameraBase::captureFrame(int timeout) {
    if (!isRunning_ || !handle_) return cv::Mat();

    std::vector<unsigned char> data(payloadSize_);
    MV_FRAME_OUT_INFO_EX frameInfo = { 0 };

    int ret = MV_CC_GetOneFrameTimeout(handle_, data.data(), payloadSize_, &frameInfo, timeout);
    if (ret != MV_OK) return cv::Mat();

    cv::Mat frame;
    if (frameInfo.enPixelType == PixelType_Gvsp_YUV422_YUYV_Packed) {
        cv::Mat yuyv(frameInfo.nHeight, frameInfo.nWidth, CV_8UC2, data.data());
        cv::cvtColor(yuyv, frame, cv::COLOR_YUV2BGR_YUY2);
    }
    else if (frameInfo.enPixelType == PixelType_Gvsp_BayerRG8) {
        cv::Mat bayer(frameInfo.nHeight, frameInfo.nWidth, CV_8UC1, data.data());
        cv::cvtColor(bayer, frame, cv::COLOR_BayerRGGB2BGR);
    }
    else if (frameInfo.enPixelType == PixelType_Gvsp_Mono8) {
        frame = cv::Mat(frameInfo.nHeight, frameInfo.nWidth, CV_8UC1, data.data());
    }
    else {
        frame = cv::Mat(frameInfo.nHeight, frameInfo.nWidth, CV_8UC3, data.data());
    }

    return frame;
}

bool CameraBase::createDirectoryIfNotExists(const std::string& dir) {
#ifdef _WIN32
    return _mkdir(dir.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

bool CameraBase::saveFrame(const cv::Mat& frame, const std::string& filename, int quality) {
    if (frame.empty()) return false;

    std::vector<int> params;
    if (filename.find(".jpg") != std::string::npos || filename.find(".jpeg") != std::string::npos) {
        params = { cv::IMWRITE_JPEG_QUALITY, quality };
    }
    else if (filename.find(".png") != std::string::npos) {
        params = { cv::IMWRITE_PNG_COMPRESSION, 3 };
    }

    return cv::imwrite(filename, frame, params);
}
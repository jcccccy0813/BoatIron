#include <iostream>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#include <errno.h>
#endif

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "MvCameraControl.h"

using namespace std;

std::atomic<bool> globalRunning(true);
std::atomic<bool> globalSave(false);
std::atomic<int> saveCount(0);
std::atomic<int> saveGroupID(0);
std::mutex saveMutex;

struct CameraHandle
{
    void* handle = nullptr;
    unsigned int index = 0;
    std::atomic<bool> isRunning{ false };
    std::atomic<bool> readyToStart{ false };
    std::string windowName;
    std::string cameraName;
};

bool CreateDirectoryIfNotExists(const std::string& dir)
{
#ifdef _WIN32
    return _mkdir(dir.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

bool SetResolution(void* handle, int width, int height)
{
    int ret = MV_CC_SetIntValue(handle, "Width", width);
    if (ret != MV_OK) return false;
    ret = MV_CC_SetIntValue(handle, "Height", height);
    return ret == MV_OK;
}

void CameraThread(CameraHandle* cam, bool isSingle = false)
{
    while (!cam->readyToStart && globalRunning)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    MV_FRAME_OUT_INFO_EX stImageInfo = { 0 };
    unsigned int nPayloadSize = 0;

    if (!SetResolution(cam->handle, 1920, 1080)) return;

    MVCC_INTVALUE stParam = { 0 };
    MV_CC_GetIntValue(cam->handle, "PayloadSize", &stParam);
    nPayloadSize = stParam.nCurValue;

    if (MV_CC_StartGrabbing(cam->handle) != MV_OK) return;

    cv::namedWindow(cam->windowName, cv::WINDOW_AUTOSIZE);

    while (globalRunning && cam->isRunning)
    {
        std::vector<unsigned char> data(nPayloadSize);
        MV_FRAME_OUT_INFO_EX frameInfo;
        int ret = MV_CC_GetOneFrameTimeout(cam->handle, data.data(), nPayloadSize, &frameInfo, 1000);
        if (ret == MV_OK)
        {
            cv::Mat frame;
            if (frameInfo.enPixelType == PixelType_Gvsp_YUV422_YUYV_Packed)
            {
                cv::Mat yuyv(frameInfo.nHeight, frameInfo.nWidth, CV_8UC2, data.data());
                cv::cvtColor(yuyv, frame, cv::COLOR_YUV2BGR_YUY2);
            }
            else if (frameInfo.enPixelType == PixelType_Gvsp_BayerRG8)
            {
                cv::Mat bayer(frameInfo.nHeight, frameInfo.nWidth, CV_8UC1, data.data());
                cv::cvtColor(bayer, frame, cv::COLOR_BayerRGGB2BGR);
            }
            else
            {
                frame = cv::Mat(frameInfo.nHeight, frameInfo.nWidth,
                    (frameInfo.enPixelType == PixelType_Gvsp_Mono8) ? CV_8UC1 : CV_8UC3, data.data());
            }

            if (!frame.empty())
            {
                cv::imshow(cam->windowName, frame);
                cv::waitKey(1);

                if (globalSave.load())
                {
                    std::lock_guard<std::mutex> lock(saveMutex);
                    int group = saveGroupID.load();
                    std::string folder = isSingle ? ((cam->cameraName == "left") ? "leftsingle" : "rightsingle") : "stereo";
                    CreateDirectoryIfNotExists(folder);

                    std::ostringstream oss;
                    oss << folder << "/" << cam->cameraName << std::setw(2) << std::setfill('0') << group << ".jpg";
                    std::string filename = oss.str();

                    if (cv::imwrite(filename, frame, { cv::IMWRITE_JPEG_QUALITY, 90 }))
                        printf("[%s] Saved: %s\n", cam->cameraName.c_str(), filename.c_str());
                    else
                        printf("[%s] Save failed!\n", cam->cameraName.c_str());

                    if (--saveCount == 0)
                    {
                        globalSave = false;
                        printf("All cameras have saved images for group %d.\n", group);
                    }
                }
            }
        }
    }

    MV_CC_StopGrabbing(cam->handle);
    cv::destroyWindow(cam->windowName);
}

void RunSingleCameraMode()
{
    int index;
    printf("Enter camera index (0 for left, 1 for right): ");
    std::cin >> index;

    MV_CC_DEVICE_INFO_LIST deviceList = { 0 };
    if (MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &deviceList) != MV_OK || index >= (int)deviceList.nDeviceNum)
    {
        printf("Invalid camera index.\n");
        return;
    }

    CameraHandle cam;
    cam.readyToStart = true;
    cam.index = index;
    cam.windowName = (index == 0) ? "left" : "right";
    cam.cameraName = cam.windowName;
    cam.isRunning = true;

    if (MV_CC_CreateHandle(&cam.handle, deviceList.pDeviceInfo[index]) != MV_OK) return;
    if (MV_CC_OpenDevice(cam.handle) != MV_OK) return;

    MV_CC_SetEnumValue(cam.handle, "TriggerMode", 0);
    MV_CC_SetFloatValue(cam.handle, "Gamma", 0.37f);

    std::thread t([&]() { CameraThread(&cam, true); });

    printf("Press 'S' to save, 'Q' to quit.\n");
    while (globalRunning)
    {
        int key = getchar();
        if (key == 's' || key == 'S')
        {
            globalSave = true;
            saveCount = 1;
            ++saveGroupID;
        }
        else if (key == 'q' || key == 'Q')
        {
            globalRunning = false;
        }
    }

    cam.isRunning = false;
    if (t.joinable()) t.join();
    MV_CC_CloseDevice(cam.handle);
    MV_CC_DestroyHandle(cam.handle);
}

void RunDualCameraMode()
{
    MV_CC_DEVICE_INFO_LIST deviceList = { 0 };
    if (MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &deviceList) != MV_OK || deviceList.nDeviceNum < 2)
    {
        printf("Need at least 2 cameras!\n");
        return;
    }

    CameraHandle cams[2];
    for (int i = 0; i < 2; ++i)
    {
        cams[i].index = i;
        cams[i].windowName = (i == 0) ? "left" : "right";
        cams[i].cameraName = cams[i].windowName;
        cams[i].isRunning = true;
        cams[i].readyToStart = true;

        if (MV_CC_CreateHandle(&cams[i].handle, deviceList.pDeviceInfo[i]) != MV_OK) return;
        if (MV_CC_OpenDevice(cams[i].handle) != MV_OK) return;

        MV_CC_SetEnumValue(cams[i].handle, "TriggerMode", 0);
        MV_CC_SetBoolValue(cams[i].handle, "GammaEnable", true);
        MV_CC_SetFloatValue(cams[i].handle, "Gamma", 0.37f);
    }

    std::thread t[2] = {
        std::thread([&]() { CameraThread(&cams[0]); }),
        std::thread([&]() { CameraThread(&cams[1]); })
    };

    printf("Press 'S' to save, 'Q' to quit.\n");
    while (globalRunning)
    {
        int key = getchar();
        if (key == 's' || key == 'S')
        {
            globalSave = true;
            saveCount = 2;
            ++saveGroupID;
        }
        else if (key == 'q' || key == 'Q')
        {
            globalRunning = false;
        }
    }

    for (int i = 0; i < 2; ++i)
    {
        cams[i].isRunning = false;
        if (t[i].joinable()) t[i].join();
        MV_CC_CloseDevice(cams[i].handle);
        MV_CC_DestroyHandle(cams[i].handle);
    }
}

int main3()
{
    int mode;
    printf("Enter mode (1 = Single Camera, 2 = Dual Camera): ");
    std::cin >> mode;

    if (mode == 1)
        RunSingleCameraMode();
    else if (mode == 2)
        RunDualCameraMode();
    else
        printf("Invalid mode.\n");

    return 0;
}

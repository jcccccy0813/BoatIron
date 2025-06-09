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
    void* handle;
    unsigned int index;
    std::atomic<bool> isRunning;
    std::atomic<bool> readyToStart; 
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
    while (!cam->readyToStart && globalRunning) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    MV_FRAME_OUT_INFO_EX stImageInfo = { 0 };
    unsigned int nPayloadSize = 0;

    if (!SetResolution(cam->handle, 1920, 1080)) return;

    MVCC_INTVALUE stParam = { 0 };
    MV_CC_GetIntValue(cam->handle, "PayloadSize", &stParam);
    nPayloadSize = stParam.nCurValue;

    int nRet = MV_CC_StartGrabbing(cam->handle);
    if (nRet != MV_OK) return;

    cv::namedWindow(cam->windowName, cv::WINDOW_AUTOSIZE);

    while (globalRunning && cam->isRunning)
    {
        std::vector<unsigned char> data(nPayloadSize);
        unsigned char* pData = data.data();

        nRet = MV_CC_GetOneFrameTimeout(cam->handle, pData, nPayloadSize, &stImageInfo, 1000);
        if (nRet == MV_OK)
        {
            cv::Mat frame;
            if (stImageInfo.enPixelType == PixelType_Gvsp_YUV422_YUYV_Packed)
            {
                cv::Mat yuyvImage(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC2, pData);
                cv::cvtColor(yuyvImage, frame, cv::COLOR_YUV2BGR_YUY2);
            }
            else if (stImageInfo.enPixelType == PixelType_Gvsp_BayerRG8)
            {
                cv::Mat bayerImage(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
                cv::cvtColor(bayerImage, frame, cv::COLOR_BayerRGGB2BGR);
            }
            else
            {
                frame = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth,
                    (stImageInfo.enPixelType == PixelType_Gvsp_Mono8) ? CV_8UC1 : CV_8UC3, pData);
            }

            if (!frame.empty())
            {
                cv::imshow(cam->windowName, frame);
                cv::waitKey(1);

                if (globalSave.load())
                {
                    std::lock_guard<std::mutex> lock(saveMutex);
                    int group = saveGroupID.load();

                    std::string folder;
                    if (isSingle)
                        folder = (cam->cameraName == "left") ? "leftsingle" : "rightsingle";
                    else
                        folder = "stereo";
                    CreateDirectoryIfNotExists(folder);

                    std::ostringstream filenameStream;
                    filenameStream << folder << "/" << cam->cameraName << std::setw(2) << std::setfill('0') << group << ".jpg";
                    std::string filename = filenameStream.str();

                    std::vector<int> compression_params = { cv::IMWRITE_JPEG_QUALITY, 90 };
                    if (cv::imwrite(filename, frame, compression_params))
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
    int cameraIndex;
    printf("Enter camera index (0 for left, 1 for right): ");
    std::cin >> cameraIndex;

    MV_CC_DEVICE_INFO_LIST stDeviceList = { 0 };
    int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if (nRet != MV_OK || cameraIndex >= (int)stDeviceList.nDeviceNum)
    {
        printf("Invalid camera index.\n");
        return;
    }

    CameraHandle cam;
    cam.readyToStart = true;

    nRet = MV_CC_CreateHandle(&cam.handle, stDeviceList.pDeviceInfo[cameraIndex]);
    if (nRet != MV_OK) return;

    nRet = MV_CC_OpenDevice(cam.handle);
    if (nRet != MV_OK) return;

   
    float gammaValue = 0.37f;
    nRet = MV_CC_SetFloatValue(cam.handle, "Gamma", gammaValue);
    if (nRet != MV_OK)
        printf("Failed to set Gamma value to %.2f on camera %d\n", gammaValue, cameraIndex);
    else
        printf("Gamma value set to %.2f on camera %d\n", gammaValue, cameraIndex);

    MV_CC_SetEnumValue(cam.handle, "TriggerMode", 0);
    cam.index = cameraIndex;
    cam.isRunning = true;
    cam.windowName = (cameraIndex == 0) ? "left" : "right";
    cam.cameraName = cam.windowName;

    std::thread camThread([&]() {
        CameraThread(&cam, true);
        });

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
    if (camThread.joinable()) camThread.join();

    MV_CC_CloseDevice(cam.handle);
    MV_CC_DestroyHandle(cam.handle);
}

void RunDualCameraMode()
{
    MV_CC_DEVICE_INFO_LIST stDeviceList = { 0 };
    int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if (nRet != MV_OK || stDeviceList.nDeviceNum < 2)
    {
        printf("Need at least 2 cameras!\n");
        return;
    }

    const int cameraNum = 2;
    CameraHandle cameras[cameraNum];
    for (int i = 0; i < cameraNum; ++i)
    {
        nRet = MV_CC_CreateHandle(&cameras[i].handle, stDeviceList.pDeviceInfo[i]);
        if (nRet != MV_OK) return;

        nRet = MV_CC_OpenDevice(cameras[i].handle);
        if (nRet != MV_OK) return;

        // 设置伽马相关
        bool gammaEnable = true;
        nRet = MV_CC_SetBoolValue(cameras[i].handle, "GammaEnable", gammaEnable);
        if (nRet != MV_OK)
            printf("Failed to enable Gamma correction on camera %d. Error: 0x%x\n", i, nRet);
        else
            printf("Gamma correction enabled on camera %d\n", i);

        float gammaValue = 0.37f;
        nRet = MV_CC_SetFloatValue(cameras[i].handle, "Gamma", gammaValue);
        if (nRet != MV_OK)
            printf("Failed to set Gamma value on camera %d. Error: 0x%x\n", i, nRet);
        else
            printf("Gamma value set to %.2f on camera %d\n", gammaValue, i);



        MV_CC_SetEnumValue(cameras[i].handle, "TriggerMode", 0);
        cameras[i].index = i;
        cameras[i].isRunning = true;
        cameras[i].windowName = (i == 0) ? "left" : "right";
        cameras[i].cameraName = cameras[i].windowName;
        cameras[i].readyToStart = true;  
    }

    std::thread threads[cameraNum];
    for (int i = 0; i < cameraNum; ++i)
    {
        threads[i] = std::thread([&, i]() {
            CameraThread(&cameras[i]);
            });
    }

    printf("Press 'S' to save, 'Q' to quit.\n");
    while (globalRunning)
    {
        int key = getchar();
        if (key == 's' || key == 'S')
        {
            globalSave = true;
            saveCount = cameraNum;
            ++saveGroupID;
        }
        else if (key == 'q' || key == 'Q')
        {
            globalRunning = false;
        }
    }

    for (auto& cam : cameras) cam.isRunning = false;
    for (auto& t : threads) if (t.joinable()) t.join();
    for (auto& cam : cameras)
    {
        MV_CC_CloseDevice(cam.handle);
        MV_CC_DestroyHandle(cam.handle);
    }
}



int main()
{
    int mode;
    printf("Enter mode (1 = Single Camera, 2 = Dual Camera): ");
    std::cin >> mode;

    switch (mode)
    {
    case 1:
        RunSingleCameraMode();
        break;
    case 2:
        RunDualCameraMode();
        break;
    default:
        printf("Invalid mode.\n");
        break;
    }

    return 0;
}

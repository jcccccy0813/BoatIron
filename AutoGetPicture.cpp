#include <windows.h>
#include <gdiplus.h>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <sstream>

#ifdef _WIN32
#include <direct.h>
#include <errno.h>
#endif

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "MvCameraControl.h"

#pragma comment(lib, "gdiplus.lib")
namespace fs = std::filesystem;
using namespace Gdiplus;

// 全局原子变量
static std::atomic<bool> globalRunning(true);
static std::atomic<bool> capturing(false);
static std::atomic<int> currentGroup(0);
static std::atomic<int> imagesCaptured(0);
static std::mutex saveMutex;
static std::condition_variable cv_capture;
static std::mutex cv_mutex;

// ====================== 投影仪相关函数 ======================

// 辅助函数：加载图片文件
std::vector<std::wstring> LoadImageFiles(const std::wstring& folder) {
    std::vector<std::wstring> files;
    try {
        for (const auto& entry : fs::directory_iterator(folder)) {
            auto ext = entry.path().extension().wstring();
            std::transform(ext.begin(), ext.end(), ext.begin(), towlower);
            if (ext == L".jpg" || ext == L".jpeg" || ext == L".png" || ext == L".bmp") {
                files.push_back(entry.path().wstring());
            }
        }
        std::sort(files.begin(), files.end());
    }
    catch (const std::exception& e) {
        MessageBoxA(nullptr, e.what(), "目录访问错误", MB_ICONERROR);
    }
    return files;
}

// 安全Graphics封装类
class SafeGraphics {
    Graphics* graphics;
public:
    explicit SafeGraphics(HDC hdc) : graphics(new Graphics(hdc)) {
        if (graphics && graphics->GetLastStatus() != Ok) {
            delete graphics;
            graphics = nullptr;
        }
    }
    ~SafeGraphics() {
        if (graphics) {
            delete graphics;
            graphics = nullptr;
        }
    }
    explicit operator bool() const { return graphics != nullptr; }
    Graphics* operator->() { return graphics; }
};

// 窗口过程
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// ====================== 相机相关函数 ======================

struct CameraHandle
{
    void* handle = nullptr;
    unsigned int index = 0;
    std::atomic<bool> isRunning{ false };
    std::atomic<bool> readyToStart{ false };
    std::string windowName;
    std::string cameraName;
    int totalImages = 0;  // 添加总图像数成员
};

static bool CreateDirectoryIfNotExists(const std::string& dir)
{
    try {
        if (!fs::exists(dir)) {
            return fs::create_directories(dir);
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "创建目录失败: " << dir << " - " << e.what() << std::endl;
        return false;
    }
}

static bool SetResolution(void* handle, int width, int height)
{
    int ret = MV_CC_SetIntValue(handle, "Width", width);
    if (ret != MV_OK) return false;
    ret = MV_CC_SetIntValue(handle, "Height", height);
    return ret == MV_OK;
}

static void CameraThread(CameraHandle* cam)
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

                // 自动保存逻辑 - 修改为保存到data/left和data/right
                if (capturing)
                {
                    if (cam->cameraName == "left" || cam->cameraName == "right")
                    {
                        std::lock_guard<std::mutex> lock(saveMutex);

                        // 创建基础目录结构
                        std::string baseDir = "data";
                        CreateDirectoryIfNotExists(baseDir);

                        // 创建相机特定目录
                        std::string cameraDir = baseDir + "/" + cam->cameraName;
                        CreateDirectoryIfNotExists(cameraDir);

                        // 生成文件名
                        std::string filename;

                        if (currentGroup == cam->totalImages - 2) {
                            // 倒数第二张是白色参考图
                            filename = cameraDir + "/white_ref.png";
                        }
                        else if (currentGroup == cam->totalImages - 1) {
                            // 最后一张是黑色参考图
                            filename = cameraDir + "/black_ref.png";
                        }
                        else {
                            // 其他图像使用两位数字序号
                            std::ostringstream oss;
                            oss << cameraDir << "/"
                                << std::setw(2) << std::setfill('0') << currentGroup << ".jpg";
                            filename = oss.str();
                        }

                        // 根据文件扩展名决定保存参数
                        std::vector<int> params;
                        if (filename.find(".jpg") != std::string::npos) {
                            params = { cv::IMWRITE_JPEG_QUALITY, 90 };
                        }
                        else {
                            params = { cv::IMWRITE_PNG_COMPRESSION, 3 }; // PNG压缩级别
                        }

                        if (cv::imwrite(filename, frame, params))
                        {
                            printf("[%s] 保存: %s\n", cam->cameraName.c_str(), filename.c_str());
                            imagesCaptured++;

                            // 通知主线程图像已保存
                            if (imagesCaptured >= 2) {
                                cv_capture.notify_one();
                            }
                        }
                    }
                }
            }
        }
    }

    MV_CC_StopGrabbing(cam->handle);
    cv::destroyWindow(cam->windowName);
}

// ====================== 同步采集函数 ======================

void RunSyncCapture() {
    // 初始化 GDI+
    ULONG_PTR token = 0;
    GdiplusStartupInput gdiplusStartupInput;
    if (GdiplusStartup(&token, &gdiplusStartupInput, nullptr) != Ok) {
        MessageBox(nullptr, L"GDI+ 初始化失败", L"错误", MB_ICONERROR);
        return;
    }

    // 加载图片
    const std::wstring imageFolder = L"graycode"; // 修改为你的图片路径
    auto images = LoadImageFiles(imageFolder);
    if (images.empty()) {
        MessageBox(nullptr, L"没有找到图片文件！", L"错误", MB_ICONERROR);
        GdiplusShutdown(token);
        return;
    }

    // 获取显示器信息
    std::vector<MONITORINFOEX> monitorList;
    EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR hMonitor, HDC, LPRECT, LPARAM lParam) -> BOOL {
        MONITORINFOEX mi = {};
        mi.cbSize = sizeof(mi);
        if (GetMonitorInfo(hMonitor, &mi)) {
            reinterpret_cast<std::vector<MONITORINFOEX>*>(lParam)->push_back(mi);
        }
        return TRUE;
        }, reinterpret_cast<LPARAM>(&monitorList));

    if (monitorList.size() < 2) {
        MessageBox(nullptr, L"未检测到第二个显示器", L"错误", MB_ICONERROR);
        GdiplusShutdown(token);
        return;
    }

    // 设置目标显示器
    const RECT& rc = monitorList[1].rcMonitor;
    int width = rc.right - rc.left;
    int height = rc.bottom - rc.top;

    // 注册窗口类
    const wchar_t CLASS_NAME[] = L"ImageSlideshowClass";
    WNDCLASS wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

    if (!RegisterClass(&wc)) {
        MessageBox(nullptr, L"窗口类注册失败", L"错误", MB_ICONERROR);
        GdiplusShutdown(token);
        return;
    }

    // 创建全屏窗口
    HWND hwnd = CreateWindowEx(
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        CLASS_NAME,
        L"投影仪幻灯片",
        WS_POPUP,
        rc.left, rc.top, width, height,
        nullptr, nullptr, GetModuleHandle(nullptr), nullptr
    );

    if (!hwnd) {
        MessageBox(nullptr, L"窗口创建失败", L"错误", MB_ICONERROR);
        GdiplusShutdown(token);
        return;
    }

    ShowWindow(hwnd, SW_SHOWNORMAL);
    UpdateWindow(hwnd);

    HDC hdcWindow = GetDC(hwnd);
    std::atomic<bool> running(true);

    // 初始化相机
    MV_CC_DEVICE_INFO_LIST deviceList = { 0 };
    if (MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &deviceList) != MV_OK || deviceList.nDeviceNum < 2)
    {
        MessageBox(nullptr, L"需要至少两个相机！", L"错误", MB_ICONERROR);
        ReleaseDC(hwnd, hdcWindow);
        DestroyWindow(hwnd);
        GdiplusShutdown(token);
        return;
    }

    CameraHandle cams[2];
    // 设置总图像数
    const int totalImages = images.size();

    for (int i = 0; i < 2; ++i)
    {
        cams[i].index = i;
        cams[i].windowName = (i == 0) ? "left" : "right";
        cams[i].cameraName = cams[i].windowName;
        cams[i].isRunning = true;
        cams[i].readyToStart = true;
        cams[i].totalImages = totalImages;  // 设置总图像数

        if (MV_CC_CreateHandle(&cams[i].handle, deviceList.pDeviceInfo[i]) != MV_OK)
        {
            printf("相机 %d 创建句柄失败\n", i);
            return;
        }
        if (MV_CC_OpenDevice(cams[i].handle) != MV_OK)
        {
            printf("相机 %d 打开设备失败\n", i);
            return;
        }

        // 设置相机参数
        MV_CC_SetEnumValue(cams[i].handle, "TriggerMode", 0); // 连续采集模式
        MV_CC_SetBoolValue(cams[i].handle, "GammaEnable", true);
        MV_CC_SetFloatValue(cams[i].handle, "Gamma", 0.37f);
        MV_CC_SetFloatValue(cams[i].handle, "ExposureTime", 10000.0f); // 曝光时间
    }

    // 启动相机线程
    std::thread cameraThreads[2] = {
        std::thread([&]() { CameraThread(&cams[0]); }),
        std::thread([&]() { CameraThread(&cams[1]); })
    };

    // 创建安全Graphics对象
    {
        SafeGraphics graphics(hdcWindow);
        if (!graphics) {
            ReleaseDC(hwnd, hdcWindow);
            DestroyWindow(hwnd);
            GdiplusShutdown(token);
            return;
        }

        graphics->SetInterpolationMode(InterpolationModeHighQualityBicubic);

        // 主消息循环和图片显示
        MSG msg;
        bool quit = false;

        for (int i = 0; i < images.size() && running && !quit; i++)
        {
            // 加载并显示图片
            std::unique_ptr<Image> pImg(Image::FromFile(images[i].c_str()));
            if (pImg && pImg->GetLastStatus() == Ok) {
                graphics->Clear(Color::Black);

                // 计算居中缩放
                UINT imgW = pImg->GetWidth();
                UINT imgH = pImg->GetHeight();
                float scale = std::min(static_cast<float>(width) / imgW,
                    static_cast<float>(height) / imgH);
                int drawW = static_cast<int>(imgW * scale);
                int drawH = static_cast<int>(imgH * scale);
                int x = (width - drawW) / 2;
                int y = (height - drawH) / 2;

                // 绘制图像
                graphics->DrawImage(pImg.get(), x, y, drawW, drawH);
            }

            // 设置当前组并触发采集
            {
                std::lock_guard<std::mutex> lock(saveMutex);
                currentGroup = i;  // 使用当前图片索引作为序号
                imagesCaptured = 0;
                capturing = true;
            }

            printf("显示图片 %d, 开始采集...\n", i + 1);

            // 等待采集完成
            {
                std::unique_lock<std::mutex> lock(cv_mutex);
                cv_capture.wait(lock, [&] {
                    return imagesCaptured >= 2 || !running;
                    });
            }

            capturing = false;
            printf("采集完成: %d\n", i + 1);

            // 处理窗口消息
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    quit = true;
                    break;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }

            if (quit) break;

            // 等待10秒后再切换到下一张图片
            printf("等待10秒...\n");
            auto start = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) {
                // 处理窗口消息
                while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    if (msg.message == WM_QUIT) {
                        quit = true;
                        break;
                    }
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
                if (quit) break;

                // 等待100毫秒
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (quit) break;
        }
    } // SafeGraphics 对象在此作用域结束时被销毁

    // 清理资源
    running = false;
    globalRunning = false;
    capturing = false;

    // 通知相机线程退出
    for (int i = 0; i < 2; i++) {
        cams[i].isRunning = false;
    }

    // 等待相机线程结束
    for (int i = 0; i < 2; i++) {
        if (cameraThreads[i].joinable()) {
            cameraThreads[i].join();
        }
        MV_CC_CloseDevice(cams[i].handle);
        MV_CC_DestroyHandle(cams[i].handle);
    }

    // 最后释放窗口资源
    ReleaseDC(hwnd, hdcWindow);
    DestroyWindow(hwnd);
    GdiplusShutdown(token);

    printf("同步采集完成！共采集 %d 组图像。\n", images.size());
}

// ====================== 主程序入口 ======================

int main() {
    // 预先创建data目录
    CreateDirectoryIfNotExists("data");

    while (true) {
        std::cout << "\n===== 结构光三维扫描系统 =====" << std::endl;
        std::cout << "1. 开始同步采集" << std::endl;
        std::cout << "0. 退出程序" << std::endl;
        std::cout << "请选择操作: ";

        int choice;
        std::cin >> choice;

        globalRunning = true; // 重置运行标志

        switch (choice) {
        case 1:
            RunSyncCapture();
            break;
        case 0:
            std::cout << "程序已退出。" << std::endl;
            return 0;
        default:
            std::cout << "无效的选择，请重新输入。" << std::endl;
            break;
        }
    }

    return 0;
}
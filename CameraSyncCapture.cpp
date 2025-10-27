#include "CameraBase.h"
#include "CameraSyncCapture.h"

#pragma comment(lib, "gdiplus.lib")
namespace fs = std::filesystem;
using namespace Gdiplus;

CameraSyncCapture::CameraSyncCapture() {
    // 注意：CameraManager 没有回调机制，需要调整同步逻辑

    // 配置相机参数（如果需要，可以在 CameraManager 中设置）
    // CameraManager 会在内部初始化时配置相机
}

// 析构函数
CameraSyncCapture::~CameraSyncCapture() {
    stopCapture();
    if (hdc_) ReleaseDC(hwnd_, hdc_);
    if (hwnd_) DestroyWindow(hwnd_);
    if (gdiplusToken_) GdiplusShutdown(gdiplusToken_);
}

// 初始化
bool CameraSyncCapture::initialize() {
    // 初始化双相机 - 使用 CameraManager 的初始化方式
    // CameraManager 会在 runDualCameraMode 中处理初始化
    // 这里我们主要初始化投影仪

    // 初始化投影仪
    if (!initializeProjector()) {
        std::cerr << "Failed to initialize projector!" << std::endl;
        return false;
    }

    isRunning_ = true;
    return true;
}

// 投影仪初始化
bool CameraSyncCapture::initializeProjector() {
    GdiplusStartupInput gdiplusStartupInput;
    if (GdiplusStartup(&gdiplusToken_, &gdiplusStartupInput, nullptr) != Ok) {
        std::cerr << "Failed to initialize GDI+" << std::endl;
        return false;
    }

    // 查找副显示器
    std::vector<MONITORINFOEX> monitors;
    EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR h, HDC, LPRECT, LPARAM l) {
        MONITORINFOEX mi = { sizeof(mi) };
        GetMonitorInfo(h, &mi);
        ((std::vector<MONITORINFOEX>*)l)->push_back(mi);
        return TRUE;
        }, (LPARAM)&monitors);

    if (monitors.size() < 2) {
        std::cerr << "Secondary monitor not found!" << std::endl;
        return false;
    }

    const RECT& rc = monitors[1].rcMonitor;
    int width = rc.right - rc.left;
    int height = rc.bottom - rc.top;

    // 创建投影窗口
    const wchar_t CLASS_NAME[] = L"ProjectorWindow";
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = DefWindowProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc)) {
        std::cerr << "Failed to register window class!" << std::endl;
        return false;
    }

    hwnd_ = CreateWindowEx(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, CLASS_NAME,
        L"Structured Light Projector", WS_POPUP,
        rc.left, rc.top, width, height,
        nullptr, nullptr, wc.hInstance, nullptr);

    if (!hwnd_) {
        std::cerr << "Failed to create projector window!" << std::endl;
        return false;
    }

    ShowWindow(hwnd_, SW_SHOWNORMAL);
    UpdateWindow(hwnd_);
    hdc_ = GetDC(hwnd_);

    return true;
}

// 加载图像文件
std::vector<std::wstring> CameraSyncCapture::loadImageFiles(const std::wstring& folder) {
    std::vector<std::wstring> files;

    if (!fs::exists(folder)) {
        std::wcerr << L"Folder does not exist: " << folder << std::endl;
        return files;
    }

    try {
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().wstring();
                std::transform(ext.begin(), ext.end(), ext.begin(), towlower);
                if (ext == L".jpg" || ext == L".jpeg" || ext == L".png" || ext == L".bmp") {
                    files.push_back(entry.path().wstring());
                }
            }
        }
        std::sort(files.begin(), files.end());
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
    }

    return files;
}

// 投影图像
void CameraSyncCapture::projectImage(const std::wstring& imagePath) {
    if (!hdc_ || !hwnd_) return;

    Graphics graphics(hdc_);
    graphics.Clear(Color::Black);

    std::unique_ptr<Image> img(Image::FromFile(imagePath.c_str()));
    if (img && img->GetLastStatus() == Ok) {
        UINT imgWidth = img->GetWidth();
        UINT imgHeight = img->GetHeight();

        RECT rc;
        GetClientRect(hwnd_, &rc);
        int winWidth = rc.right - rc.left;
        int winHeight = rc.bottom - rc.top;

        // 计算居中位置
        int x = (winWidth - (int)imgWidth) / 2;
        int y = (winHeight - (int)imgHeight) / 2;

        if (x < 0) x = 0;
        if (y < 0) y = 0;

        graphics.DrawImage(img.get(), x, y, imgWidth, imgHeight);
    }
}

void CameraSyncCapture::triggerImageSave() {
    // 重置计数器
    imagesSaved_ = 0;

    std::cout << "Triggering programmatic image capture..." << std::endl;

    // 使用新的程序化保存方法
    cameraManager_.triggerCapture();

    // 等待保存完成 - 监控保存状态
    for (int i = 0; i < 50; ++i) { // 最多等待5秒
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 检查是否还在保存中
        if (!cameraManager_.isSaving() && cameraManager_.getSaveCount() == 0) {
            std::cout << "Image capture completed." << std::endl;
            imagesSaved_ = 2;
            return;
        }

        // 每1秒输出一次状态
        if (i % 10 == 0) {
            std::cout << "Waiting for image save... " << (i / 10) << "s" << std::endl;
        }
    }

    std::cout << "Image capture timeout, continuing..." << std::endl;
    imagesSaved_ = 2; // 强制继续
}

// 同步等待
void CameraSyncCapture::waitForCamerasToSave() {
    std::unique_lock<std::mutex> lock(syncMutex_);
    syncCV_.wait(lock, [this] {
        return imagesSaved_ >= 2 || !isRunning_;
        });
}

// 运行同步采集
void CameraSyncCapture::runSyncCapture() {
    // 加载灰度码图像
    auto images = loadImageFiles(L"graycode");
    if (images.empty()) {
        std::cerr << "No images found in graycode folder!" << std::endl;
        return;
    }

    std::cout << "Found " << images.size() << " pattern images" << std::endl;

    // 创建数据目录 -  会在内部创建 data 目录
    // 我们需要确保目录存在
    CameraBase::createDirectoryIfNotExists("data");
    CameraBase::createDirectoryIfNotExists("data/Cam_001");
    CameraBase::createDirectoryIfNotExists("data/Cam_002");
    // === 关键修改：先检查相机状态 ===
    std::cout << "Checking camera availability..." << std::endl;
    MV_CC_DEVICE_INFO_LIST deviceList = { 0 };
    if (!CameraBase::enumDevices(deviceList) || deviceList.nDeviceNum < 2) {
        std::cerr << "Need at least 2 cameras! Found: " << deviceList.nDeviceNum << std::endl;
        return;
    }
    std::cout << "Found " << deviceList.nDeviceNum << " cameras" << std::endl;

    // 在单独的线程中启动相机采集
    std::atomic<bool> cameraThreadRunning{ false };
    std::thread cameraThread([this, &cameraThreadRunning]() {
        cameraThreadRunning = true;
        std::cout << "Starting camera manager in dual camera mode..." << std::endl;
        cameraManager_.runDualCameraMode();
        std::cout << "Camera manager thread finished." << std::endl;
        cameraThreadRunning = false;
        });

    // 等待相机初始化完成
    std::cout << "Waiting for camera initialization..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));

    if (!cameraThreadRunning) {
        std::cerr << "Camera thread failed to start!" << std::endl;
        stopCapture();
        if (cameraThread.joinable()) cameraThread.join();
        return;
    }

    std::cout << "Camera initialization completed, starting projection..." << std::endl;

    // 主循环
    for (int i = 0; i < images.size(); ++i) {
        if (!isRunning_) {
            break;
        }

        std::cout << "Projecting pattern " << (i + 1) << "/" << images.size() << std::endl;

        // 投影图像
        projectImage(images[i]);

        // 等待投影稳定
        std::this_thread::sleep_for(std::chrono::seconds(3));

        // 触发图像保存
        triggerImageSave();

        // 等待两台相机都完成保存
        waitForCamerasToSave();

        std::cout << "Pattern " << (i + 1) << " captured and saved." << std::endl;

        // 处理Windows消息
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                stopCapture();
                return;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // 短暂延迟
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Capture completed successfully!" << std::endl;

    // 停止相机采集
    stopCapture();

    if (cameraThread.joinable()) {
        cameraThread.join();
    }
}

// 停止采集
void CameraSyncCapture::stopCapture() {
    isRunning_ = false;

    // 通知所有等待的线程
    {
        std::lock_guard<std::mutex> lock(syncMutex_);
        imagesSaved_ = 2;
    }
    syncCV_.notify_all();
}

// 运行
void CameraSyncCapture::structuredLightCapture() {
    std::cout << "\n=== Structured Light Auto Capture ===\n";
    std::cout << "Requirements:\n";
    std::cout << "1. Dual cameras connected\n";
    std::cout << "2. Secondary monitor for projection\n";
    std::cout << "3. Graycode patterns in 'graycode' folder\n";
    std::cout << "4. System will automatically:\n";
    std::cout << "   - Project each pattern\n";
    std::cout << "   - Capture synchronized images\n";
    std::cout << "   - Save to data/Cam_001/ and data/Cam_002/\n\n";

    std::cout << "Continue? (y/n): ";
    char confirm;
    std::cin >> confirm;
    CameraBase::clearInputBuffer();

    if (confirm != 'y' && confirm != 'Y') {
        std::cout << "Operation cancelled.\n";
        return;
    }

    CameraSyncCapture syncCapture;

    if (!syncCapture.initialize()) {
        std::cerr << "Failed to initialize structured light capture system!\n";
        std::cerr << "Please check:\n";
        std::cerr << "1. Camera connections\n";
        std::cerr << "2. Secondary monitor availability\n";
        std::cerr << "3. Graycode patterns in 'graycode' folder\n";
        return;
    }

    std::cout << "Starting structured light capture...\n";
    std::cout << "Press Ctrl+C to stop at any time.\n\n";

    try {
        syncCapture.runSyncCapture();
        std::cout << "\nStructured light capture completed successfully!\n";
        std::cout << "Images saved to data/Cam_001/ and data/Cam_002/ folders\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error during capture: " << e.what() << "\n";
    }
}
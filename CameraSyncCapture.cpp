#include "CameraSyncCapture.h"


#pragma comment(lib, "gdiplus.lib")
namespace fs = std::filesystem;
using namespace Gdiplus;

CameraSyncCapture::CameraSyncCapture() {
    // 设置图像保存回调
    cameraCapture_.setImageSavedCallback(
        [this](const std::string& cameraName, const std::string& filename) {
            this->handleImageSaved(cameraName, filename);
        }
    );

    // 配置相机参数
    CameraBase::CameraConfig leftConfig;
    leftConfig.windowName = "Left Camera";
    leftConfig.cameraName = "left";
    leftConfig.exposureTime = 10000.0f;

    CameraBase::CameraConfig rightConfig;
    rightConfig.windowName = "Right Camera";
    rightConfig.cameraName = "right";
    rightConfig.exposureTime = 10000.0f;
}

CameraSyncCapture::~CameraSyncCapture() {
    stopCapture();
    if (hdc_) ReleaseDC(hwnd_, hdc_);
    if (hwnd_) DestroyWindow(hwnd_);
    if (gdiplusToken_) GdiplusShutdown(gdiplusToken_);
}

bool CameraSyncCapture::initialize() {
    // 初始化双相机
    if (!cameraCapture_.initDualCameras()) {
        std::cerr << "Failed to initialize dual cameras!" << std::endl;
        return false;
    }

    // 初始化投影仪
    if (!initializeProjector()) {
        std::cerr << "Failed to initialize projector!" << std::endl;
        return false;
    }

    return true;
}

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

void CameraSyncCapture::handleImageSaved(const std::string& cameraName, const std::string& filename) {
    imagesSaved_++;
    std::cout << "[" << cameraName << "] Image saved: " << filename << std::endl;

    if (imagesSaved_ >= 2) {
        syncCV_.notify_one();
    }
}

void CameraSyncCapture::waitForCamerasToSave() {
    std::unique_lock<std::mutex> lock(syncMutex_);
    syncCV_.wait(lock, [this] {
        return imagesSaved_ >= 2 || !cameraCapture_.isRunning();
        });
}

void CameraSyncCapture::runSyncCapture() {
    // 加载灰度码图像
    auto images = loadImageFiles(L"graycode");
    if (images.empty()) {
        std::cerr << "No images found in graycode folder!" << std::endl;
        return;
    }

    std::cout << "Found " << images.size() << " pattern images" << std::endl;

    // 创建数据目录
    CameraBase::createDirectoryIfNotExists("data");
    CameraBase::createDirectoryIfNotExists("data/left");
    CameraBase::createDirectoryIfNotExists("data/right");

    // 启动相机采集
    cameraCapture_.startCapture();

    // 主循环
    for (int i = 0; i < images.size(); ++i) {
        if (!cameraCapture_.isRunning()) {
            break;
        }

        std::cout << "Projecting pattern " << (i + 1) << "/" << images.size() << std::endl;

        // 投影图像
        projectImage(images[i]);

        // 等待投影稳定
        std::this_thread::sleep_for(std::chrono::seconds(3));

        // 重置计数器并开始采集
        {
            std::lock_guard<std::mutex> lock(syncMutex_);
            imagesSaved_ = 0;
            currentGroup_ = i;
        }

        // 触发相机保存
        cameraCapture_.saveImages(i, false);

        // 等待两台相机都完成保存
        waitForCamerasToSave();

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
    std::cout << "Left images: " << cameraCapture_.getLeftImages().size() << std::endl;
    std::cout << "Right images: " << cameraCapture_.getRightImages().size() << std::endl;
}

void CameraSyncCapture::stopCapture() {
    cameraCapture_.stopCapture();

    // 通知所有等待的线程
    {
        std::lock_guard<std::mutex> lock(syncMutex_);
        imagesSaved_ = 2;
    }
    syncCV_.notify_all();
}
void CameraSyncCapture::structuredLightCapture() {
    std::cout << "\n=== Structured Light Auto Capture ===\n";
    std::cout << "Requirements:\n";
    std::cout << "1. Dual cameras connected\n";
    std::cout << "2. Secondary monitor for projection\n";
    std::cout << "3. Graycode patterns in 'graycode' folder\n";
    std::cout << "4. System will automatically:\n";
    std::cout << "   - Project each pattern\n";
    std::cout << "   - Capture synchronized images\n";
    std::cout << "   - Save to data/left/ and data/right/\n\n";

    std::cout << "Continue? (y/n): ";
    char confirm;
    std::cin >> confirm;
    clearInputBuffer();

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
        std::cout << "Images saved to data/left/ and data/right/ folders\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error during capture: " << e.what() << "\n";
    }
}
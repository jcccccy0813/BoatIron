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

// ȫ��ԭ�ӱ���
static std::atomic<bool> globalRunning(true);
static std::atomic<bool> capturing(false);
static std::atomic<int> currentGroup(0);
static std::atomic<int> imagesCaptured(0);
static std::mutex saveMutex;
static std::condition_variable cv_capture;
static std::mutex cv_mutex;

// ====================== ͶӰ����غ��� ======================

// ��������������ͼƬ�ļ�
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
        MessageBoxA(nullptr, e.what(), "Ŀ¼���ʴ���", MB_ICONERROR);
    }
    return files;
}

// ��ȫGraphics��װ��
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

// ���ڹ���
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// ====================== �����غ��� ======================

struct CameraHandle
{
    void* handle = nullptr;
    unsigned int index = 0;
    std::atomic<bool> isRunning{ false };
    std::atomic<bool> readyToStart{ false };
    std::string windowName;
    std::string cameraName;
    int totalImages = 0;  // �����ͼ������Ա
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
        std::cerr << "����Ŀ¼ʧ��: " << dir << " - " << e.what() << std::endl;
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

                // �Զ������߼� - �޸�Ϊ���浽data/left��data/right
                if (capturing)
                {
                    if (cam->cameraName == "left" || cam->cameraName == "right")
                    {
                        std::lock_guard<std::mutex> lock(saveMutex);

                        // ��������Ŀ¼�ṹ
                        std::string baseDir = "data";
                        CreateDirectoryIfNotExists(baseDir);

                        // ��������ض�Ŀ¼
                        std::string cameraDir = baseDir + "/" + cam->cameraName;
                        CreateDirectoryIfNotExists(cameraDir);

                        // �����ļ���
                        std::string filename;

                        if (currentGroup == cam->totalImages - 2) {
                            // �����ڶ����ǰ�ɫ�ο�ͼ
                            filename = cameraDir + "/white_ref.png";
                        }
                        else if (currentGroup == cam->totalImages - 1) {
                            // ���һ���Ǻ�ɫ�ο�ͼ
                            filename = cameraDir + "/black_ref.png";
                        }
                        else {
                            // ����ͼ��ʹ����λ�������
                            std::ostringstream oss;
                            oss << cameraDir << "/"
                                << std::setw(2) << std::setfill('0') << currentGroup << ".jpg";
                            filename = oss.str();
                        }

                        // �����ļ���չ�������������
                        std::vector<int> params;
                        if (filename.find(".jpg") != std::string::npos) {
                            params = { cv::IMWRITE_JPEG_QUALITY, 90 };
                        }
                        else {
                            params = { cv::IMWRITE_PNG_COMPRESSION, 3 }; // PNGѹ������
                        }

                        if (cv::imwrite(filename, frame, params))
                        {
                            printf("[%s] ����: %s\n", cam->cameraName.c_str(), filename.c_str());
                            imagesCaptured++;

                            // ֪ͨ���߳�ͼ���ѱ���
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

// ====================== ͬ���ɼ����� ======================

void RunSyncCapture() {
    // ��ʼ�� GDI+
    ULONG_PTR token = 0;
    GdiplusStartupInput gdiplusStartupInput;
    if (GdiplusStartup(&token, &gdiplusStartupInput, nullptr) != Ok) {
        MessageBox(nullptr, L"GDI+ ��ʼ��ʧ��", L"����", MB_ICONERROR);
        return;
    }

    // ����ͼƬ
    const std::wstring imageFolder = L"graycode"; // �޸�Ϊ���ͼƬ·��
    auto images = LoadImageFiles(imageFolder);
    if (images.empty()) {
        MessageBox(nullptr, L"û���ҵ�ͼƬ�ļ���", L"����", MB_ICONERROR);
        GdiplusShutdown(token);
        return;
    }

    // ��ȡ��ʾ����Ϣ
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
        MessageBox(nullptr, L"δ��⵽�ڶ�����ʾ��", L"����", MB_ICONERROR);
        GdiplusShutdown(token);
        return;
    }

    // ����Ŀ����ʾ��
    const RECT& rc = monitorList[1].rcMonitor;
    int width = rc.right - rc.left;
    int height = rc.bottom - rc.top;

    // ע�ᴰ����
    const wchar_t CLASS_NAME[] = L"ImageSlideshowClass";
    WNDCLASS wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

    if (!RegisterClass(&wc)) {
        MessageBox(nullptr, L"������ע��ʧ��", L"����", MB_ICONERROR);
        GdiplusShutdown(token);
        return;
    }

    // ����ȫ������
    HWND hwnd = CreateWindowEx(
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        CLASS_NAME,
        L"ͶӰ�ǻõ�Ƭ",
        WS_POPUP,
        rc.left, rc.top, width, height,
        nullptr, nullptr, GetModuleHandle(nullptr), nullptr
    );

    if (!hwnd) {
        MessageBox(nullptr, L"���ڴ���ʧ��", L"����", MB_ICONERROR);
        GdiplusShutdown(token);
        return;
    }

    ShowWindow(hwnd, SW_SHOWNORMAL);
    UpdateWindow(hwnd);

    HDC hdcWindow = GetDC(hwnd);
    std::atomic<bool> running(true);

    // ��ʼ�����
    MV_CC_DEVICE_INFO_LIST deviceList = { 0 };
    if (MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &deviceList) != MV_OK || deviceList.nDeviceNum < 2)
    {
        MessageBox(nullptr, L"��Ҫ�������������", L"����", MB_ICONERROR);
        ReleaseDC(hwnd, hdcWindow);
        DestroyWindow(hwnd);
        GdiplusShutdown(token);
        return;
    }

    CameraHandle cams[2];
    // ������ͼ����
    const int totalImages = images.size();

    for (int i = 0; i < 2; ++i)
    {
        cams[i].index = i;
        cams[i].windowName = (i == 0) ? "left" : "right";
        cams[i].cameraName = cams[i].windowName;
        cams[i].isRunning = true;
        cams[i].readyToStart = true;
        cams[i].totalImages = totalImages;  // ������ͼ����

        if (MV_CC_CreateHandle(&cams[i].handle, deviceList.pDeviceInfo[i]) != MV_OK)
        {
            printf("��� %d �������ʧ��\n", i);
            return;
        }
        if (MV_CC_OpenDevice(cams[i].handle) != MV_OK)
        {
            printf("��� %d ���豸ʧ��\n", i);
            return;
        }

        // �����������
        MV_CC_SetEnumValue(cams[i].handle, "TriggerMode", 0); // �����ɼ�ģʽ
        MV_CC_SetBoolValue(cams[i].handle, "GammaEnable", true);
        MV_CC_SetFloatValue(cams[i].handle, "Gamma", 0.37f);
        MV_CC_SetFloatValue(cams[i].handle, "ExposureTime", 10000.0f); // �ع�ʱ��
    }

    // ��������߳�
    std::thread cameraThreads[2] = {
        std::thread([&]() { CameraThread(&cams[0]); }),
        std::thread([&]() { CameraThread(&cams[1]); })
    };

    // ������ȫGraphics����
    {
        SafeGraphics graphics(hdcWindow);
        if (!graphics) {
            ReleaseDC(hwnd, hdcWindow);
            DestroyWindow(hwnd);
            GdiplusShutdown(token);
            return;
        }

        graphics->SetInterpolationMode(InterpolationModeHighQualityBicubic);

        // ����Ϣѭ����ͼƬ��ʾ
        MSG msg;
        bool quit = false;

        for (int i = 0; i < images.size() && running && !quit; i++)
        {
            // ���ز���ʾͼƬ
            std::unique_ptr<Image> pImg(Image::FromFile(images[i].c_str()));
            if (pImg && pImg->GetLastStatus() == Ok) {
                graphics->Clear(Color::Black);

                // �����������
                UINT imgW = pImg->GetWidth();
                UINT imgH = pImg->GetHeight();
                float scale = std::min(static_cast<float>(width) / imgW,
                    static_cast<float>(height) / imgH);
                int drawW = static_cast<int>(imgW * scale);
                int drawH = static_cast<int>(imgH * scale);
                int x = (width - drawW) / 2;
                int y = (height - drawH) / 2;

                // ����ͼ��
                graphics->DrawImage(pImg.get(), x, y, drawW, drawH);
            }

            // ���õ�ǰ�鲢�����ɼ�
            {
                std::lock_guard<std::mutex> lock(saveMutex);
                currentGroup = i;  // ʹ�õ�ǰͼƬ������Ϊ���
                imagesCaptured = 0;
                capturing = true;
            }

            printf("��ʾͼƬ %d, ��ʼ�ɼ�...\n", i + 1);

            // �ȴ��ɼ����
            {
                std::unique_lock<std::mutex> lock(cv_mutex);
                cv_capture.wait(lock, [&] {
                    return imagesCaptured >= 2 || !running;
                    });
            }

            capturing = false;
            printf("�ɼ����: %d\n", i + 1);

            // ��������Ϣ
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    quit = true;
                    break;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }

            if (quit) break;

            // �ȴ�10������л�����һ��ͼƬ
            printf("�ȴ�10��...\n");
            auto start = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) {
                // ��������Ϣ
                while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    if (msg.message == WM_QUIT) {
                        quit = true;
                        break;
                    }
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
                if (quit) break;

                // �ȴ�100����
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (quit) break;
        }
    } // SafeGraphics �����ڴ����������ʱ������

    // ������Դ
    running = false;
    globalRunning = false;
    capturing = false;

    // ֪ͨ����߳��˳�
    for (int i = 0; i < 2; i++) {
        cams[i].isRunning = false;
    }

    // �ȴ�����߳̽���
    for (int i = 0; i < 2; i++) {
        if (cameraThreads[i].joinable()) {
            cameraThreads[i].join();
        }
        MV_CC_CloseDevice(cams[i].handle);
        MV_CC_DestroyHandle(cams[i].handle);
    }

    // ����ͷŴ�����Դ
    ReleaseDC(hwnd, hdcWindow);
    DestroyWindow(hwnd);
    GdiplusShutdown(token);

    printf("ͬ���ɼ���ɣ����ɼ� %d ��ͼ��\n", images.size());
}

// ====================== ��������� ======================

int main() {
    // Ԥ�ȴ���dataĿ¼
    CreateDirectoryIfNotExists("data");

    while (true) {
        std::cout << "\n===== �ṹ����άɨ��ϵͳ =====" << std::endl;
        std::cout << "1. ��ʼͬ���ɼ�" << std::endl;
        std::cout << "0. �˳�����" << std::endl;
        std::cout << "��ѡ�����: ";

        int choice;
        std::cin >> choice;

        globalRunning = true; // �������б�־

        switch (choice) {
        case 1:
            RunSyncCapture();
            break;
        case 0:
            std::cout << "�������˳���" << std::endl;
            return 0;
        default:
            std::cout << "��Ч��ѡ�����������롣" << std::endl;
            break;
        }
    }

    return 0;
}
#include <windows.h>
#include <gdiplus.h>
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>

#pragma comment(lib, "gdiplus.lib")
namespace fs = std::filesystem;
using namespace Gdiplus;

std::vector<std::wstring> LoadImageFiles(const std::wstring& folder) {
    std::vector<std::wstring> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        auto ext = entry.path().extension().wstring();
        std::transform(ext.begin(), ext.end(), ext.begin(), towlower);
        if (ext == L".jpg" || ext == L".jpeg" || ext == L".png" || ext == L".bmp") {
            files.push_back(entry.path().wstring());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) PostQuitMessage(0);
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

int main() {
    // ��ʼ�� GDI+
    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR token;
    GdiplusStartup(&token, &gdiplusStartupInput, nullptr);

    // ����ͼƬ�ļ���·�������滻��
    std::wstring imageFolder = L"D:\\project\\BoatIron\\graycode";
    auto images = LoadImageFiles(imageFolder);
    if (images.empty()) {
        MessageBox(nullptr, L"û���ҵ�ͼƬ�ļ���", L"����", MB_ICONERROR);
        return 1;
    }

    // ��ȡ�ڶ�����ʾ����Ϣ
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
        MessageBox(nullptr, L"δ��⵽�ڶ�����ʾ����", L"����", MB_ICONERROR);
        return 1;
    }

    RECT rc = monitorList[1].rcMonitor;
    int width = rc.right - rc.left;
    int height = rc.bottom - rc.top;

    // ע�ᴰ����
    const wchar_t CLASS_NAME[] = L"ImageSlideshowClass";
    WNDCLASS wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = CLASS_NAME;
    RegisterClass(&wc);

    // ����ȫ���ޱ߿򴰿�
    HWND hwnd = CreateWindowEx(
        WS_EX_TOPMOST,
        CLASS_NAME,
        L"Slideshow",
        WS_POPUP,
        rc.left, rc.top, width, height,
        nullptr, nullptr, GetModuleHandle(nullptr), nullptr
    );

    if (!hwnd) {
        MessageBox(nullptr, L"���ڴ���ʧ�ܣ�", L"����", MB_ICONERROR);
        return 1;
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    HDC hdcWindow = GetDC(hwnd);
    Graphics graphics(hdcWindow);
    graphics.SetInterpolationMode(InterpolationModeHighQualityBicubic);

    size_t index = 0;
    bool running = true;

    // ��ʾͼƬ�߳�
    std::thread slideshowThread([&]() {
        while (running) {
            Image img(images[index].c_str());
            graphics.Clear(Color::Black);

            UINT iw = img.GetWidth();
            UINT ih = img.GetHeight();
            float scale = std::min((float)width / iw, (float)height / ih);
            int drawW = static_cast<int>(iw * scale);
            int drawH = static_cast<int>(ih * scale);
            int x = (width - drawW) / 2;
            int y = (height - drawH) / 2;
            graphics.DrawImage(&img, x, y, drawW, drawH);

            index = (index + 1) % images.size();
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
        });

    // ��Ϣѭ��
    MSG msg = {};
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    running = false;
    slideshowThread.join();
    ReleaseDC(hwnd, hdcWindow);
    GdiplusShutdown(token);
    return 0;
}

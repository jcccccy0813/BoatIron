#include "Common.h"
#include "CameraCapture.h"
#include "Calibration.h"
#include "StereoCalibration.h"
#include "StereoMatching.h"
#include "CameraSyncCapture.h"  
#include "GrayCodeDecoder.h"
#include "GrayCodeGenerator.h"
#include "ImageListGenerator.h"

int main() {
    int choice = 0;
    bool running = true;

    while (running) {
        // 打印主菜单
        printMainMenu();
        // 获取用户输入
        if (!(std::cin >> choice)) {
            clearInputBuffer();
            std::cerr << "Invalid input. Please enter a number.\n";
            continue;
        }
        // 处理用户选择
        switch (choice) {
        // 1. 拍照
        case 1: {
            int mode;
            std::cout << "\n1. Single Camera\n";
            std::cout << "2. Dual Cameras\n";
            std::cout << "Enter capture mode (1-2): ";

            if (!(std::cin >> mode) || (mode != 1 && mode != 2)) {
                clearInputBuffer();
                std::cerr << "Invalid mode selection.\n";
                continue;  // 直接继续主循环
            }

            if (mode == 1) CameraCapture::captureSingleCamera();
            else CameraCapture::captureDualCameras();
            break;
        }
         // 2. 投影拍照同步，结构光
        case 2:
            CameraSyncCapture::structuredLightCapture();
            break;
            // 3. 单相机标定
        case 3:
            Calibration::calibrateLeftCamera();
            break;
            // 4. 双相机标定
        case 4:
            Calibration::calibrateRightCamera();
            break;
            // 5. 双相机标定
        case 5:
            StereoCalibration::stereoCalibration();
            break;
             // 6. 双相机匹配
        case 6:
            StereoMatching::stereoMatching();
            break;
            // 7. 格雷码生成
        case 7:
            GrayCodeGenerator::generateGrayCodePatterns();
            break;
            // 8. 格雷码解码
        case 8:
            GrayCodeDecoder::decodeGrayCodePatterns();
            break;
            // 9. 图片列表生成
        case 9:
            ImageListGenerator::generateImageList();
            break;
            // 10. 退出
        case 10:
            std::cout << "Exiting program.\n";
            running = false;
            break;
            // 其他选项
        default:
            std::cerr << "Invalid choice. Please enter 1-10.\n";
        }
        // 运行完一个选项后，等待用户输入回车键，继续主循环
        if (choice != 10) {
            std::cout << "\nPress Enter to return to main menu...";
            clearInputBuffer();
            std::cin.get();
        }
    }
    return 0;
}
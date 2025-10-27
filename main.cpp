#include "Common.h"
#include "CameraManager.h"
#include "Calibration.h"
#include "StereoCalibration.h"
#include "StereoMatching.h"
#include "CameraSyncCapture.h"  
#include "GrayCodeDecoder.h"
#include "GrayCodeGenerator.h"
#include "ImageListGenerator.h"
#include <iostream>
#include <limits>



// 安全的等待回车函数
void waitForEnter() {
    std::cout << "Press Enter to continue...";
    clearInputBuffer();
    std::cin.get();
}

int main() {
    int choice = 0;
    bool running = true;

    // 创建 CameraManager 实例
    CameraManager cameraManager;

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
            std::cout << "\n=== Camera Capture ===\n";
            std::cout << "1. Single Camera\n";
            std::cout << "2. Dual Cameras\n";
            std::cout << "Enter capture mode (1-2): ";

            if (!(std::cin >> mode) || (mode != 1 && mode != 2)) {
                clearInputBuffer();
                std::cerr << "Invalid mode selection.\n";
                continue;
            }

            // 使用 CameraManager
            if (mode == 1) {
                std::cout << "Starting single camera mode...\n";
                cameraManager.runSingleCameraMode();
            }
            else {
                std::cout << "Starting dual camera mode...\n";
                cameraManager.runDualCameraMode();
            }
            break;
        }
              // 2. 投影拍照同步，结构光
        case 2:
            std::cout << "Starting structured light capture...\n";
            CameraSyncCapture::structuredLightCapture();
            break;
            // 3. 左相机标定
        case 3:
            std::cout << "Starting left camera calibration...\n";
            Calibration::calibrateLeftCamera();
            break;
            // 4. 右相机标定
        case 4:
            std::cout << "Starting right camera calibration...\n";
            Calibration::calibrateRightCamera();
            break;
            // 5. 双目标定
        case 5:
            std::cout << "Starting stereo calibration...\n";
            StereoCalibration::stereoCalibration();
            break;
            // 6. 双目匹配
        case 6:
            std::cout << "Starting stereo matching...\n";
            StereoMatching::stereoMatching();
            break;
            // 7. 格雷码生成
        case 7:
            std::cout << "Generating gray code patterns...\n";
            GrayCodeGenerator::generateGrayCodePatterns();
            break;
            // 8. 格雷码解码
        case 8:
            std::cout << "Decoding gray code patterns...\n";
            GrayCodeDecoder::decodeGrayCodePatterns();
            break;
            // 9. 图片列表生成
        case 9:
            std::cout << "Generating image list...\n";
            ImageListGenerator::generateImageList();
            break;
            // 10. 退出
        case 10:
            std::cout << "Exiting program. Goodbye!\n";
            running = false;
            break;
            // 其他选项
        default:
            std::cerr << "Invalid choice. Please enter 1-10.\n";
        }

        // 运行完一个选项后，等待用户输入回车键，继续主循环
        if (choice != 10 && running) {
            waitForEnter();
        }
    }

    return 0;
}
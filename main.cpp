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
        printMainMenu();

        if (!(std::cin >> choice)) {
            clearInputBuffer();
            std::cerr << "Invalid input. Please enter a number.\n";
            continue;
        }

        switch (choice) {
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
        case 2:
            CameraSyncCapture::structuredLightCapture();
            break;
        case 3:
            Calibration::calibrateLeftCamera();
            break;
        case 4:
            Calibration::calibrateRightCamera();
            break;
        case 5:
            StereoCalibration::stereoCalibration();
            break;
        case 6:
            StereoMatching::stereoMatching();
            break;
        case 7:
            GrayCodeGenerator::generateGrayCodePatterns();
            break;
        case 8:
            GrayCodeDecoder::decodeGrayCodePatterns();
            break;
        case 9:
            ImageListGenerator::generateImageList();
            break;
        case 10:
            std::cout << "Exiting program.\n";
            running = false;
            break;
        default:
            std::cerr << "Invalid choice. Please enter 1-10.\n";
        }

        if (choice != 10) {
            std::cout << "\nPress Enter to return to main menu...";
            clearInputBuffer();
            std::cin.get();
        }
    }
    return 0;
}
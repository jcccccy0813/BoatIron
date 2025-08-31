#include "StereoCalibration.h"


StereoCalibration::PresetConfig StereoCalibration::getPresetConfig(PresetType preset) {
    // 获得初始配置
    PresetConfig config;
    // 设置参数
    switch (preset) {
    case PRESET_CHARUCO_9X6_25MM:
        config.boardSize = cv::Size(9, 6);
        config.pattern = CHARUCOBOARD;
        config.squareSize = 25.0f;
        config.markerSize = 12.5f;
        config.arucoDict = cv::aruco::DICT_4X4_50;
        break;

    case PRESET_CHESSBOARD_11X8_20MM:
    default:
        config.boardSize = cv::Size(11, 8);
        config.pattern = CHESSBOARD;
        config.squareSize = 20.0f;
        config.markerSize = 0.0f;
        config.arucoDict = cv::aruco::DICT_4X4_50;
    }
    return config;
}
// 基于预设标定
bool StereoCalibration::calibrateWithPreset(const std::string& imageListFile,
    const std::string& leftIntrinsicFile,
    const std::string& rightIntrinsicFile,
    PresetType preset) {
    PresetConfig config = getPresetConfig(preset);

    std::vector<cv::String> imageList;
    if (!readStringList(imageListFile, imageList)) {
        std::cerr << "Error: Failed to read image list file\n";
        return false;
    }

    std::cout << "\n=== Stereo Calibration with Preset ===\n";
    std::cout << "Board size: " << config.boardSize.width << "x" << config.boardSize.height << "\n";
    std::cout << "Pattern: " << (config.pattern == CHESSBOARD ? "Chessboard" : "ChArUco") << "\n";
    std::cout << "Square size: " << config.squareSize << "mm\n";
    std::cout << "Left intrinsic: " << leftIntrinsicFile << "\n";
    std::cout << "Right intrinsic: " << rightIntrinsicFile << "\n";

    if (!calibrate(imageList, config.boardSize, config.pattern,
        config.squareSize, config.markerSize,
        config.arucoDict, leftIntrinsicFile, rightIntrinsicFile)) {
        return false;
    }

    return saveCalibrationResults("intrinsics.yml", "extrinsics.yml");
}
// 保存标定结果
bool StereoCalibration::saveCalibrationResults(const std::string& intrinsicsFile,
    const std::string& extrinsicsFile) {
    // Save intrinsic parameters
    cv::FileStorage fs(intrinsicsFile, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open " << intrinsicsFile << " for writing\n";
        return false;
    }

    time_t now = time(0);
    char* dt = ctime(&now);

    fs << "calibration_time" << dt;
    fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0];
    fs << "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
    fs.release();

    // Save extrinsic parameters
    cv::FileStorage extrinsics(extrinsicsFile, cv::FileStorage::WRITE);
    if (!extrinsics.isOpened()) {
        std::cerr << "Error: Could not open " << extrinsicsFile << " for writing\n";
        return false;
    }

    extrinsics << "calibration_time" << dt;
    extrinsics << "R" << stereoParams_.R << "T" << stereoParams_.T;
    extrinsics << "R1" << stereoParams_.R1 << "R2" << stereoParams_.R2;
    extrinsics << "P1" << stereoParams_.P1 << "P2" << stereoParams_.P2;
    extrinsics << "Q" << stereoParams_.Q;
    extrinsics.release();

    return true;
}

bool StereoCalibration::calibrate(const std::vector<cv::String>& imagelist,
    cv::Size boardSize, PatternType pattern,
    float squareSize, float markerSize,
    cv::aruco::PredefinedDictionaryType arucoDict,
    const std::string& leftIntrinsicFile,
    const std::string& rightIntrinsicFile,
    bool displayCorners, bool showRectified) {
    if (imagelist.size() % 2 != 0) {
        std::cerr << "Error: Image list must contain even number of images\n";
        return false;
    }

    const int maxScale = 2;
    std::vector<std::vector<cv::Point2f>> imagePoints[2];
    std::vector<std::vector<cv::Point3f>> objectPoints;
    cv::Size imageSize;

    int nimages = (int)imagelist.size() / 2;
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    std::vector<std::string> goodImageList;

    cv::Size boardSizeInnerCorners, boardSizeUnits;
    if (pattern == CHESSBOARD) {
        boardSizeInnerCorners = boardSize;
        boardSizeUnits.height = boardSize.height + 1;
        boardSizeUnits.width = boardSize.width + 1;
    }
    else if (pattern == CHARUCOBOARD) {
        boardSizeUnits = boardSize;
        boardSizeInnerCorners.width = boardSize.width - 1;
        boardSizeInnerCorners.height = boardSize.height - 1;
    }
    else {
        std::cerr << "Error: Unsupported pattern type\n";
        return false;
    }

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(arucoDict);
    cv::aruco::CharucoBoard ch_board(boardSizeUnits, squareSize, markerSize, dictionary);
    cv::aruco::CharucoDetector ch_detector(ch_board);
    std::vector<int> markerIds;

    // Load intrinsic parameters
    {
        cv::FileStorage fs(leftIntrinsicFile, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Error: Failed to open " << leftIntrinsicFile << std::endl;
            return false;
        }
        fs["camera_matrix"] >> cameraMatrix[0];
        fs["distortion_coefficients"] >> distCoeffs[0];
        fs.release();
    }
    {
        cv::FileStorage fs(rightIntrinsicFile, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Error: Failed to open " << rightIntrinsicFile << std::endl;
            return false;
        }
        fs["camera_matrix"] >> cameraMatrix[1];
        fs["distortion_coefficients"] >> distCoeffs[1];
        fs.release();
    }
    int i, j = 0, k;
    // Detect corners in all images
    for ( i = 0, j = 0; i < nimages; i++) {
        for ( k = 0; k < 2; k++) {
            const std::string& filename = imagelist[i * 2 + k];
            cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Warning: Could not load image " << filename << std::endl;
                break;
            }

            if (imageSize == cv::Size()) {
                imageSize = img.size();
            }
            else if (img.size() != imageSize) {
                std::cerr << "Warning: Image " << filename << " has different size. Skipping pair\n";
                break;
            }

            bool found = false;
            std::vector<cv::Point2f>& corners = imagePoints[k][j];
            for (int scale = 1; scale <= maxScale; scale++) {
                cv::Mat timg;
                if (scale == 1)
                    timg = img;
                else
                    cv::resize(img, timg, cv::Size(), scale, scale, cv::INTER_LINEAR_EXACT);

                if (pattern == CHESSBOARD) {
                    found = cv::findChessboardCorners(timg, boardSizeInnerCorners, corners,
                        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
                }
                else if (pattern == CHARUCOBOARD) {
                    ch_detector.detectBoard(timg, corners, markerIds);
                    found = corners.size() == (size_t)(boardSizeInnerCorners.height * boardSizeInnerCorners.width);
                }

                if (found) {
                    if (scale > 1) {
                        cv::Mat cornersMat(corners);
                        cornersMat *= 1. / scale;
                    }
                    break;
                }
            }

            if (displayCorners) {
                std::cout << filename << std::endl;
                cv::Mat cimg, cimg1;
                cv::cvtColor(img, cimg, cv::COLOR_GRAY2BGR);
                cv::drawChessboardCorners(cimg, boardSizeInnerCorners, corners, found);
                double sf = 640. / std::max(img.rows, img.cols);
                cv::resize(cimg, cimg1, cv::Size(), sf, sf, cv::INTER_LINEAR_EXACT);
                cv::imshow("corners", cimg1);
                char c = (char)cv::waitKey(500);
                if (c == 27 || c == 'q' || c == 'Q')
                    return false;
            }
            else {
                std::cout << ".";
            }

            if (!found) {
                std::cerr << "Warning: Could not find corners in " << filename << std::endl;
                break;
            }

            if (pattern == CHESSBOARD) {
                cv::cornerSubPix(img, corners, cv::Size(11, 11), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            }
        }
        if (k == 2) {
            goodImageList.push_back(imagelist[i * 2]);
            goodImageList.push_back(imagelist[i * 2 + 1]);
            j++;
        }
    }
    std::cout << "\n" << j << " pairs have been successfully detected.\n";
    nimages = j;
    if (nimages < 2) {
        std::cerr << "Error: Need at least 2 valid image pairs for calibration\n";
        return false;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for (int i = 0; i < nimages; i++) {
        for (int j = 0; j < boardSizeInnerCorners.height; j++) {
            for (int k = 0; k < boardSizeInnerCorners.width; k++) {
                objectPoints[i].push_back(cv::Point3f(k * squareSize, j * squareSize, 0));
            }
        }
    }

    std::cout << "Running stereo calibration...\n";
    double rms = cv::stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
        cameraMatrix[0], distCoeffs[0],
        cameraMatrix[1], distCoeffs[1],
        imageSize, stereoParams_.R, stereoParams_.T, stereoParams_.E, stereoParams_.F,
        cv::CALIB_FIX_INTRINSIC,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
    std::cout << "Stereo calibration RMS error = " << rms << std::endl;

    // Calculate epipolar error
    double err = 0;
    int npoints = 0;
    std::vector<cv::Vec3f> lines[2];
    for (int i = 0; i < nimages; i++) {
        int npt = (int)imagePoints[0][i].size();
        cv::Mat imgpt[2];
        for (int k = 0; k < 2; k++) {
            imgpt[k] = cv::Mat(imagePoints[k][i]);
            cv::undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
            cv::computeCorrespondEpilines(imgpt[k], k + 1, stereoParams_.F, lines[k]);
        }
        for (int j = 0; j < npt; j++) {
            double errij = fabs(imagePoints[0][i][j].x * lines[1][j][0] +
                imagePoints[0][i][j].y * lines[1][j][1] + lines[1][j][2]) +
                fabs(imagePoints[1][i][j].x * lines[0][j][0] +
                    imagePoints[1][i][j].y * lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    std::cout << "Average epipolar error = " << err / npoints << std::endl;

    // Stereo rectification
    cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
        cameraMatrix[1], distCoeffs[1],
        imageSize, stereoParams_.R, stereoParams_.T,
        stereoParams_.R1, stereoParams_.R2,
        stereoParams_.P1, stereoParams_.P2,
        stereoParams_.Q,
        cv::CALIB_ZERO_DISPARITY, 1, imageSize);

    return true;
}
//读取列表
bool StereoCalibration::readStringList(const std::string& filename, std::vector<cv::String>& l) {
    l.clear();
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    cv::FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != cv::FileNode::SEQ) {
        std::cerr << "File is not a sequence\n";
        return false;
    }

    for (cv::FileNodeIterator it = n.begin(); it != n.end(); ++it) {
        l.push_back((std::string)*it);
    }
    return !l.empty();
}
//运行
void StereoCalibration::stereoCalibration() {
    StereoCalibration stereoCalib;
    std::cout << "\n=== Stereo Calibration ===\n";
    std::cout << "Using preset: Chessboard 11x8, 20mm squares\n";
    std::cout << "Requirements:\n";
    std::cout << "1. Pre-calibrated left_camera.yml\n";
    std::cout << "2. Pre-calibrated right_camera.yml\n";
    std::cout << "3. stereo_list.xml with image pairs\n";

    if (stereoCalib.calibrateWithPreset("stereo_list.xml")) {
        std::cout << "\nStereo calibration successful!\n";
        std::cout << "Output files:\n";
        std::cout << "- intrinsics.yml (combined camera parameters)\n";
        std::cout << "- extrinsics.yml (R, T, R1, R2, P1, P2, Q)\n";
        std::cout << "Use these for stereo matching\n";
    }
    else {
        std::cerr << "\nStereo calibration failed! Possible reasons:\n";
        std::cerr << "1. Missing/Mismatched camera calibration files\n";
        std::cerr << "2. Invalid stereo image pairs\n";
        std::cerr << "3. Inconsistent chessboard detection\n";
    }
}
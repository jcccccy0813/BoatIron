#include "Calibration.h"


Calibration::Calibration() {}
bool Calibration::calibrateSelectedCamera(CameraType cameraType, PresetType preset) {
    std::string imageListFile;
    std::string outputFile;

    switch (cameraType) {
    case LEFT_CAMERA:
        imageListFile = "left_list.xml";
        outputFile = "left_camera.yml";
        std::cout << "开始标定左相机..." << std::endl;
        break;
    case RIGHT_CAMERA:
        imageListFile = "right_list.xml";
        outputFile = "right_camera.yml";
        std::cout << "开始标定右相机..." << std::endl;
        break;
    default:
        std::cerr << "Error: Invalid camera type!" << std::endl;
        return false;
    }

    return calibrateWithPreset(imageListFile, outputFile, preset);
}

Calibration::PresetConfig Calibration::getPresetConfig(PresetType preset) {
    PresetConfig config;
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
        config.markerSize = 0.0f; // 棋盘格不需要markerSize
        config.arucoDict = cv::aruco::DICT_4X4_50;
    }
    return config;
}

bool Calibration::calibrateWithPreset(const std::string& imageListFile,
    const std::string& outputFile,
    PresetType preset) {

    PresetConfig config = getPresetConfig(preset);

    std::vector<std::string> imageList;
    if (!readStringList(imageListFile, imageList) || imageList.empty()) {
        std::cerr << "Error: Failed to read image list or list is empty\n";
        return false;
    }

    return calibrateSingleCamera(imageList, config.boardSize, config.pattern,
        config.squareSize, config.markerSize,
        config.arucoDict, outputFile);
}

bool Calibration::calibrateSingleCamera(const std::vector<std::string>& imageList,
    cv::Size boardSize, PatternType pattern,
    float squareSize, float markerSize,
    cv::aruco::PredefinedDictionaryType arucoDict,
    const std::string& outputFile) {
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize;
    int nImages = (int)imageList.size();

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(arucoDict);
    cv::aruco::CharucoBoard ch_board(boardSize, squareSize, markerSize, dictionary);
    std::vector<int> markerIds;
    cv::aruco::CharucoDetector ch_detector(ch_board);

    for (int i = 0; i < nImages; i++) {
        const std::string& filename = imageList[i];
        cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Could not load image: " << filename << std::endl;
            continue;
        }

        if (imageSize == cv::Size()) {
            imageSize = img.size();
        }
        else if (img.size() != imageSize) {
            std::cerr << "Image " << filename << " has different size. Skipping." << std::endl;
            continue;
        }

        std::vector<cv::Point2f> pointBuf;
        cv::Mat viewGray;
        cv::cvtColor(img, viewGray, cv::COLOR_BGR2GRAY);

        bool found = false;
        switch (pattern) {
        case CHESSBOARD:
            found = cv::findChessboardCorners(viewGray, boardSize, pointBuf,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
            if (found) {
                cv::cornerSubPix(viewGray, pointBuf, cv::Size(11, 11),
                    cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            }
            break;

        case CHARUCOBOARD:
            ch_detector.detectBoard(img, pointBuf, markerIds);
            found = pointBuf.size() == (size_t)((boardSize.width - 1) * (boardSize.height - 1));
            break;

        case CIRCLES_GRID:
            found = cv::findCirclesGrid(img, boardSize, pointBuf);
            break;

        case ASYMMETRIC_CIRCLES_GRID:
            found = cv::findCirclesGrid(img, boardSize, pointBuf, cv::CALIB_CB_ASYMMETRIC_GRID);
            break;
        }

        if (found) {
            imagePoints.push_back(pointBuf);

            if (pattern != CHARUCOBOARD) {
                cv::drawChessboardCorners(img, boardSize, cv::Mat(pointBuf), found);
            }
            else {
                cv::drawChessboardCorners(img, cv::Size(boardSize.width - 1, boardSize.height - 1),
                    cv::Mat(pointBuf), found);
            }

            cv::imshow("Calibration", img);
            cv::waitKey(100);
        }
    }

    if (imagePoints.size() < 3) {
        std::cerr << "Not enough valid images for calibration (need at least 3)" << std::endl;
        return false;
    }

    cameraParams_.imageSize = imageSize;
    float gridWidth = squareSize * (pattern != CHARUCOBOARD ? (boardSize.width - 1) : (boardSize.width - 2));
    bool releaseObject = false;

    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<float> reprojErrs;
    std::vector<cv::Point3f> newObjPoints;
    double totalAvgErr = 0;

    bool ok = runCalibration(imagePoints, imageSize, boardSize, pattern, squareSize, 1.0f,
        gridWidth, releaseObject, 0, cameraParams_.cameraMatrix,
        cameraParams_.distCoeffs, rvecs, tvecs, reprojErrs,
        newObjPoints, totalAvgErr);

    if (ok) {
        cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);
        if (fs.isOpened()) {
            time_t tt;
            time(&tt);
            struct tm* t2 = localtime(&tt);
            char buf[1024];
            strftime(buf, sizeof(buf) - 1, "%c", t2);

            fs << "calibration_time" << buf;
            fs << "image_width" << imageSize.width;
            fs << "image_height" << imageSize.height;
            fs << "board_width" << boardSize.width;
            fs << "board_height" << boardSize.height;
            fs << "square_size" << squareSize;
            fs << "camera_matrix" << cameraParams_.cameraMatrix;
            fs << "distortion_coefficients" << cameraParams_.distCoeffs;
            fs << "avg_reprojection_error" << totalAvgErr;

            if (!reprojErrs.empty()) {
                fs << "per_view_reprojection_errors" << cv::Mat(reprojErrs);
            }

            fs.release();
            return true;
        }
    }

    return false;
}

bool Calibration::runCalibration(const std::vector<std::vector<cv::Point2f>>& imagePoints,
    cv::Size imageSize, cv::Size boardSize, PatternType pattern,
    float squareSize, float aspectRatio, float gridWidth,
    bool releaseObject, int flags, cv::Mat& cameraMatrix,
    cv::Mat& distCoeffs, std::vector<cv::Mat>& rvecs,
    std::vector<cv::Mat>& tvecs, std::vector<float>& reprojErrs,
    std::vector<cv::Point3f>& newObjPoints, double& totalAvgErr) {
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    if (flags & cv::CALIB_FIX_ASPECT_RATIO) {
        cameraMatrix.at<double>(0, 0) = aspectRatio;
    }

    distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    std::vector<std::vector<cv::Point3f>> objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], pattern);

    int offset = pattern != CHARUCOBOARD ? boardSize.width - 1 : boardSize.width - 2;
    objectPoints[0][offset].x = objectPoints[0][0].x + gridWidth;
    newObjPoints = objectPoints[0];

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    int iFixedPoint = -1;
    if (releaseObject) {
        iFixedPoint = boardSize.width - 1;
    }

    double rms = cv::calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
        cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
        flags | cv::CALIB_USE_LU);
    std::cout << "RMS error reported by calibrateCamera: " << rms << std::endl;

    bool ok = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);

    if (releaseObject) {
        std::cout << "New board corners: " << std::endl;
        std::cout << newObjPoints[0] << std::endl;
        std::cout << newObjPoints[boardSize.width - 1] << std::endl;
        std::cout << newObjPoints[boardSize.width * (boardSize.height - 1)] << std::endl;
        std::cout << newObjPoints.back() << std::endl;
    }

    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
        rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

void Calibration::calcChessboardCorners(cv::Size boardSize, float squareSize,
    std::vector<cv::Point3f>& corners, PatternType patternType) {
    corners.resize(0);

    switch (patternType) {
    case CHESSBOARD:
    case CIRCLES_GRID:
        for (int i = 0; i < boardSize.height; i++) {
            for (int j = 0; j < boardSize.width; j++) {
                corners.push_back(cv::Point3f(float(j * squareSize),
                    float(i * squareSize), 0));
            }
        }
        break;

    case ASYMMETRIC_CIRCLES_GRID:
        for (int i = 0; i < boardSize.height; i++) {
            for (int j = 0; j < boardSize.width; j++) {
                corners.push_back(cv::Point3f(float((2 * j + i % 2) * squareSize),
                    float(i * squareSize), 0));
            }
        }
        break;

    case CHARUCOBOARD:
        for (int i = 0; i < boardSize.height - 1; i++) {
            for (int j = 0; j < boardSize.width - 1; j++) {
                corners.push_back(cv::Point3f(float(j * squareSize),
                    float(i * squareSize), 0));
            }
        }
        break;
    }
}

double Calibration::computeReprojectionErrors(const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    std::vector<float>& perViewErrors) {
    std::vector<cv::Point2f> imagePoints2;
    int totalPoints = 0;
    double totalErr = 0;
    perViewErrors.resize(objectPoints.size());

    for (size_t i = 0; i < objectPoints.size(); i++) {
        cv::projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i],
            cameraMatrix, distCoeffs, imagePoints2);
        double err = cv::norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), cv::NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

bool Calibration::readStringList(const std::string& filename, std::vector<std::string>& l) {
    l.resize(0);
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    size_t dir_pos = filename.rfind('/');
    if (dir_pos == std::string::npos) {
        dir_pos = filename.rfind('\\');
    }

    cv::FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != cv::FileNode::SEQ) {
        return false;
    }

    cv::FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        std::string fname = (std::string)*it;
        if (dir_pos != std::string::npos) {
            std::string fpath = cv::samples::findFile(filename.substr(0, dir_pos + 1) + fname, false);
            if (fpath.empty()) {
                fpath = cv::samples::findFile(fname);
            }
            fname = fpath;
        }
        else {
            fname = cv::samples::findFile(fname);
        }
        l.push_back(fname);
    }
    return true;
}
void Calibration::calibrateLeftCamera() {
    Calibration calib;
    std::cout << "\n=== Left Camera Calibration ===\n";
    std::cout << "Using preset: Chessboard 11x8, 20mm squares\n";
    std::cout << "Loading images from left_list.xml...\n";

    // 使用新的 calibrateSelectedCamera 方法
    if (calib.calibrateSelectedCamera(Calibration::LEFT_CAMERA)) {
        std::cout << "\nCalibration successful!\n";
        std::cout << "Results saved to left_camera.yml\n";
        std::cout << "Use this file for stereo calibration\n";
    }
    else {
        std::cerr << "\nCalibration failed! Possible reasons:\n";
        std::cerr << "1. Invalid/missing image files in left_list.xml\n";
        std::cerr << "2. Incorrect chessboard pattern settings\n";
        std::cerr << "3. Not enough valid images (need at least 3)\n";
    }
}

void Calibration::calibrateRightCamera() {
    Calibration calib;
    std::cout << "\n=== Right Camera Calibration ===\n";
    std::cout << "Using preset: Chessboard 11x8, 20mm squares\n";
    std::cout << "Loading images from right_list.xml...\n";

    // 使用新的 calibrateSelectedCamera 方法
    if (calib.calibrateSelectedCamera(Calibration::RIGHT_CAMERA)) {
        std::cout << "\nCalibration successful!\n";
        std::cout << "Results saved to right_camera.yml\n";
        std::cout << "Use this file for stereo calibration\n";
    }
    else {
        std::cerr << "\nCalibration failed! Possible reasons:\n";
        std::cerr << "1. Invalid/missing image files in right_list.xml\n";
        std::cerr << "2. Incorrect chessboard pattern settings\n";
        std::cerr << "3. Not enough valid images (need at least 3)\n";
    }
}
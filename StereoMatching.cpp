#include "StereoMatching.h"

// StereoMatching
StereoMatching::StereoMatching() {
    // 设置默认参数
    params_.algorithm = STEREO_SGBM;
    params_.maxDisparity = 176;
    params_.colorDisplay = true;
    params_.imageList = "stereo_pairs.txt";
    params_.intrinsicFile = "intrinsics.yml";
    params_.extrinsicFile = "extrinsics.yml";
    params_.disparityOutput = "disparity";
    params_.pointCloudOutput = "pointcloud";
}
//进程主函数
bool StereoMatching::process(
    const std::string& imageList,
    const std::string& intrinsicFile,
    const std::string& extrinsicFile,
    const std::string& disparityOutput,
    const std::string& pointCloudOutput,
    StereoAlgorithm algorithm,
    int maxDisparity,
    bool colorDisplay)
{
    // 更新参数
    params_.imageList = imageList;
    params_.intrinsicFile = intrinsicFile;
    params_.extrinsicFile = extrinsicFile;
    params_.disparityOutput = disparityOutput;
    params_.pointCloudOutput = pointCloudOutput;
    params_.algorithm = algorithm;
    params_.maxDisparity = maxDisparity;
    params_.colorDisplay = colorDisplay;

    // 验证必要参数
    if (params_.imageList.empty()) {
        std::cerr << "Error: Image list file not specified!" << std::endl;
        return false;
    }

    // 读取图像列表
    std::ifstream file(params_.imageList);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open image list file: " << params_.imageList << std::endl;
        return false;
    }

    std::vector<std::string> imagePaths;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            imagePaths.push_back(line);
        }
    }

    if (imagePaths.size() < 2) {
        std::cerr << "Error: Need at least 2 images in the list" << std::endl;
        return false;
    }

    // 加载图像
    cv::Mat img1 = cv::imread(imagePaths[0], params_.colorDisplay ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(imagePaths[1], params_.colorDisplay ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images!" << std::endl;
        return false;
    }

    // 加载相机参数
    cv::Mat cameraMatrix[2], distCoeffs[2];
    cv::Mat R, T, R1, R2, P1, P2, Q;

    {
        cv::FileStorage fs(params_.intrinsicFile, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Error: Failed to open intrinsic file" << std::endl;
            return false;
        }
        fs["M1"] >> cameraMatrix[0];
        fs["D1"] >> distCoeffs[0];
        fs["M2"] >> cameraMatrix[1];
        fs["D2"] >> distCoeffs[1];
    }

    {
        cv::FileStorage fs(params_.extrinsicFile, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Error: Failed to open extrinsic file" << std::endl;
            return false;
        }
        fs["R"] >> R;
        fs["T"] >> T;
        fs["R1"] >> R1;
        fs["R2"] >> R2;
        fs["P1"] >> P1;
        fs["P2"] >> P2;
        fs["Q"] >> Q;
    }

    // 校正图像
    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, img1.size(), CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, img1.size(), CV_16SC2, map21, map22);

    cv::Mat img1r, img2r;
    cv::remap(img1, img1r, map11, map12, cv::INTER_LINEAR);
    cv::remap(img2, img2r, map21, map22, cv::INTER_LINEAR);

    // 创建匹配器
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);

    // 根据算法类型初始化
    if (params_.algorithm == STEREO_BM) {
        initStereoBM(bm, params_.maxDisparity);
    }
    else {
        initStereoSGBM(sgbm, params_.maxDisparity, img1r.channels());
    }

    // 计算视差
    cv::Mat disp, disp8;
    int64 t = cv::getTickCount();

    if (params_.algorithm == STEREO_BM) {
        bm->compute(img1r, img2r, disp);
    }
    else {
        sgbm->compute(img1r, img2r, disp);
    }

    t = cv::getTickCount() - t;
    std::cout << "Time elapsed: " << t * 1000 / cv::getTickFrequency() << "ms" << std::endl;

    // 转换视差格式
    if (params_.algorithm != STEREO_BM) {
        disp.convertTo(disp8, CV_8U, 255 / (params_.maxDisparity * 16.0));
    }
    else {
        disp.convertTo(disp8, CV_8U, 255 / (params_.maxDisparity * 1.0));
    }

    // 保存结果
    if (!params_.disparityOutput.empty()) {
        std::string disparityFile = params_.disparityOutput + ".png";
        cv::imwrite(disparityFile, disp8);
        std::cout << "Disparity map saved to: " << disparityFile << std::endl;
    }

    if (!params_.pointCloudOutput.empty() && !Q.empty()) {
        cv::Mat xyz;
        cv::Mat floatDisp;
        disp.convertTo(floatDisp, CV_32F, 1.0 / 16.0);
        cv::reprojectImageTo3D(floatDisp, xyz, Q, true);

        std::string cloudFile = params_.pointCloudOutput + ".xyz";
        saveColoredXYZ(cloudFile, xyz, img1r);
        std::cout << "Point cloud saved to: " << cloudFile << std::endl;
    }

    return true;
}
// 初始化BM算法
void StereoMatching::initStereoBM(cv::Ptr<cv::StereoBM>& bm, int maxDisparity) {
    bm->setPreFilterCap(31);
    bm->setBlockSize(15);
    bm->setMinDisparity(0);
    bm->setNumDisparities(maxDisparity);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);
}
// 初始化SGBM算法
void StereoMatching::initStereoSGBM(cv::Ptr<cv::StereoSGBM>& sgbm, int maxDisparity, int channels) {
    sgbm->setPreFilterCap(63);
    sgbm->setBlockSize(5);
    sgbm->setP1(8 * channels * 5 * 5);
    sgbm->setP2(32 * channels * 5 * 5);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(maxDisparity);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);

    switch (params_.algorithm) {
    case STEREO_HH: sgbm->setMode(cv::StereoSGBM::MODE_HH); break;
    case STEREO_HH4: sgbm->setMode(cv::StereoSGBM::MODE_HH4); break;
    case STEREO_3WAY: sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY); break;
    default: sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
    }
}
// 保存点云
void StereoMatching::saveColoredXYZ(const std::string& filename,
    const cv::Mat& mat,
    const cv::Mat& color_img) {
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename.c_str(), "wt");
    if (!fp) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }

    cv::Mat color_rgb;
    if (color_img.channels() == 1) {
        cv::cvtColor(color_img, color_rgb, cv::COLOR_GRAY2RGB);
    }
    else {
        color_rgb = color_img.clone();
    }

    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            cv::Vec3f point = mat.at<cv::Vec3f>(y, x);
            if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;

            cv::Vec3b color = color_rgb.at<cv::Vec3b>(y, x);
            fprintf(fp, "%f %f %f %d %d %d\n",
                point[0], point[1], point[2],
                color[2], color[1], color[0]); // RGB order
        }
    }
    fclose(fp);
}
// 打印帮助信息
void StereoMatching::printHelp() {
    std::cout << "\nStereo Matching Parameters:\n";
    std::cout << "imageList: Path to image list file (default: stereo_pairs.txt)\n";
    std::cout << "intrinsicFile: Intrinsic parameters file (default: intrinsics.yml)\n";
    std::cout << "extrinsicFile: Extrinsic parameters file (default: extrinsics.yml)\n";
    std::cout << "disparityOutput: Disparity map output prefix (default: disparity)\n";
    std::cout << "pointCloudOutput: Point cloud output prefix (default: pointcloud)\n";
    std::cout << "algorithm: Stereo algorithm (0=BM,1=SGBM,2=HH,3=HH4,4=3WAY, default:1)\n";
    std::cout << "maxDisparity: Maximum disparity value (default:176)\n";
    std::cout << "colorDisplay: Use color images (default:true)\n";
}

// 运行
void StereoMatching::stereoMatching() {
    std::cout << "\n=== Stereo Matching ===\n";
    std::cout << "Using default parameters:\n";
    std::cout << "- Algorithm: SGBM\n";
    std::cout << "- Max disparity: 176\n";
    std::cout << "- Input files: stereo_pairs.txt\n";
    std::cout << "- Calibration files: intrinsics.yml & extrinsics.yml\n";
    std::cout << "- Output: disparity.png & pointcloud.xyz\n\n";

    StereoMatching matcher;
    if (matcher.process("stereo_pairs.txt",  // 图像列表
        "intrinsics.yml",    // 内参文件
        "extrinsics.yml",    // 外参文件
        "disparity",         // 视差图前缀
        "pointcloud")) {     // 点云前缀
        std::cout << "\nResults saved to:\n";
        std::cout << "- disparity.png\n";
        std::cout << "- pointcloud.xyz\n";
    }
    else {
        std::cerr << "\nStereo matching failed!\n";
    }
}
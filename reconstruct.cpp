#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

int main7()
{
    // === 1. 读取内参和外参 ===
    cv::Mat M1, M2, D1, D2, R, T;
    cv::FileStorage fs1("intrinsics.yml", cv::FileStorage::READ);
    fs1["M1"] >> M1;
    fs1["D1"] >> D1;
    fs1["M2"] >> M2;
    fs1["D2"] >> D2;
    fs1.release();

    cv::FileStorage fs2("extrinsics.yml", cv::FileStorage::READ);
    fs2["R"] >> R;
    fs2["T"] >> T;
    fs2.release();

    // === 2. 读取像素坐标图 ===
    cv::Mat xL = cv::imread("x_left.exr", cv::IMREAD_UNCHANGED);  // float32
    cv::Mat yL = cv::imread("y_left.exr", cv::IMREAD_UNCHANGED);
    cv::Mat xR = cv::imread("x_right.exr", cv::IMREAD_UNCHANGED);

    if (xL.empty() || xR.empty() || yL.empty())
    {
        std::cerr << "Error loading EXR files." << std::endl;
        return -1;
    }

    // === 3. 构造投影矩阵 ===
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    M1.copyTo(P1(cv::Rect(0, 0, 3, 3)));  // 左相机投影矩阵 [M1 | 0]

    cv::Mat Rt;
    cv::hconcat(R, T, Rt);               // [R | T]
    cv::Mat P2 = M2 * Rt;                // 右相机投影矩阵 M2*[R|T]

    // === 4. 打开输出文件 ===
    std::ofstream xyzFile("triangulated.xyz");
    if (!xyzFile.is_open())
    {
        std::cerr << "Error opening output file." << std::endl;
        return -1;
    }

    // === 5. 遍历图像像素，逐点三角测量 ===
    int width = xL.cols;
    int height = xL.rows;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float xl = xL.at<float>(y, x);
            float xr = xR.at<float>(y, x);
            float yl = yL.at<float>(y, x);

            // 跳过非法匹配点（例如无值或视差为负）
            if (!cv::checkRange(xl) || !cv::checkRange(xr) || std::abs(xl - xr) < 0.01)
                continue;

            // 构造对应点
            cv::Mat pts_4d;
            cv::Mat pl = (cv::Mat_<double>(2, 1) << xl, yl);
            cv::Mat pr = (cv::Mat_<double>(2, 1) << xr, yl);

            cv::Mat ptsL = (cv::Mat_<double>(2, 1) << xl, yl);
            cv::Mat ptsR = (cv::Mat_<double>(2, 1) << xr, yl);

            cv::Mat point4D;
            cv::triangulatePoints(P1, P2, ptsL, ptsR, point4D);

            // 齐次坐标归一化
            float X = point4D.at<float>(0, 0) / point4D.at<float>(3, 0);
            float Y = point4D.at<float>(1, 0) / point4D.at<float>(3, 0);
            float Z = point4D.at<float>(2, 0) / point4D.at<float>(3, 0);

            if (!cv::checkRange(cv::Vec3f(X, Y, Z))) continue;

            xyzFile << X << " " << Y << " " << Z << "\n";
        }
    }

    xyzFile.close();
    std::cout << "点云生成完成：triangulated.xyz" << std::endl;
    return 0;
}

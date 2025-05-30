/*
 *  stereo_match.cpp
 *  calibration
 *
 *  Created by Victor  Eruhimov on 1/18/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <sstream>
#include <iostream>     // for std::cout, std::cerr
#include <fstream>      // for std::ifstream
#include <string>   

using namespace cv;
using namespace std;

static void print_help(char** argv)
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: %s <left_image> <right_image> [--algorithm=bm|sgbm|hh|hh4|sgbm3way] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
           "[--no-display] [--color] [-o=<disparity_image>] [-p=<point_cloud_file>]\n", argv[0]);
}

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

int main(int argc, char** argv)
{
    // 初始化命令行参数
    cv::CommandLineParser parser(argc, argv,
        "{help h||}{list||}{algorithm|sgbm|}{max-disparity|64|}{blocksize|5|}"
        "{no-display||}{color||}{scale|1|}{i||}{e||}{o||}{p||}");

    if (parser.has("help")) {
        print_help(argv);
        return 0;
    }

    std::string list_file = parser.get<std::string>("list");
    std::string intrinsic_filename = parser.get<std::string>("i");
    std::string extrinsic_filename = parser.get<std::string>("e");
    std::string disparity_filename = parser.get<std::string>("o");
    std::string point_cloud_filename = parser.get<std::string>("p");
    std::string algorithm = parser.get<std::string>("algorithm");

    int numberOfDisparities = parser.get<int>("max-disparity");
    int SADWindowSize = parser.get<int>("blocksize");
    float scale = parser.get<float>("scale");
    bool no_display = parser.has("no-display");
    bool color_display = parser.has("color");

    if (list_file.empty()) {
        std::cerr << "Error: Please provide --list=<image_list.txt>" << std::endl;
        return -1;
    }

    enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3, STEREO_3WAY = 4, STEREO_HH4 = 5 };
    int alg = algorithm == "bm" ? STEREO_BM :
        algorithm == "sgbm" ? STEREO_SGBM :
        algorithm == "hh" ? STEREO_HH :
        algorithm == "var" ? STEREO_VAR :
        algorithm == "hh4" ? STEREO_HH4 :
        algorithm == "sgbm3way" ? STEREO_3WAY : -1;

    if (alg < 0) {
        std::cerr << "Unknown algorithm: " << algorithm << std::endl;
        return -1;
    }

    Ptr<StereoBM> bm = StereoBM::create(16, 9);
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

    std::ifstream infile(list_file);
    if (!infile.is_open()) {
        std::cerr << "Failed to open list file: " << list_file << std::endl;
        return -1;
    }

    std::string left_path, right_path;
    int pair_idx = 0;
    while (infile >> left_path >> right_path) {
        std::cout << "\n[INFO] Processing pair #" << ++pair_idx << ": " << left_path << " " << right_path << std::endl;

        Mat img1 = imread(left_path, alg == STEREO_BM ? IMREAD_GRAYSCALE : IMREAD_COLOR);
        Mat img2 = imread(right_path, alg == STEREO_BM ? IMREAD_GRAYSCALE : IMREAD_COLOR);
        if (img1.empty() || img2.empty()) {
            std::cerr << "Could not load image pair: " << left_path << ", " << right_path << std::endl;
            continue;
        }

        if (scale != 1.f) {
            resize(img1, img1, Size(), scale, scale);
            resize(img2, img2, Size(), scale, scale);
        }

        Size img_size = img1.size();
        Rect roi1, roi2;
        Mat Q;

        if (!intrinsic_filename.empty() && !extrinsic_filename.empty()) {
            FileStorage fs(intrinsic_filename, FileStorage::READ);
            if (!fs.isOpened()) {
                std::cerr << "Failed to open intrinsic file." << std::endl;
                return -1;
            }
            Mat M1, D1, M2, D2;
            fs["M1"] >> M1; fs["D1"] >> D1;
            fs["M2"] >> M2; fs["D2"] >> D2;
            M1 *= scale; M2 *= scale;

            fs.open(extrinsic_filename, FileStorage::READ);
            if (!fs.isOpened()) {
                std::cerr << "Failed to open extrinsic file." << std::endl;
                return -1;
            }
            Mat R, T, R1, P1, R2, P2;
            fs["R"] >> R; fs["T"] >> T;

            stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q,
                CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

            Mat map11, map12, map21, map22;
            initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
            initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

            Mat img1r, img2r;
            remap(img1, img1r, map11, map12, INTER_LINEAR);
            remap(img2, img2r, map21, map22, INTER_LINEAR);
            img1 = img1r; img2 = img2r;
        }

        numberOfDisparities = (numberOfDisparities > 0) ? numberOfDisparities :
            ((img_size.width / 8) + 15) & -16;

        bm->setPreFilterCap(31);
        bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
        bm->setMinDisparity(0);
        bm->setNumDisparities(numberOfDisparities);
        bm->setTextureThreshold(10);
        bm->setUniquenessRatio(15);
        bm->setSpeckleWindowSize(100);
        bm->setSpeckleRange(32);
        bm->setDisp12MaxDiff(1);

        int cn = img1.channels();
        int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
        sgbm->setPreFilterCap(63);
        sgbm->setBlockSize(sgbmWinSize);
        sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
        sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
        sgbm->setMinDisparity(0);
        sgbm->setNumDisparities(numberOfDisparities);
        sgbm->setUniquenessRatio(10);
        sgbm->setSpeckleWindowSize(100);
        sgbm->setSpeckleRange(32);
        sgbm->setDisp12MaxDiff(1);

        if (alg == STEREO_HH)
            sgbm->setMode(StereoSGBM::MODE_HH);
        else if (alg == STEREO_SGBM)
            sgbm->setMode(StereoSGBM::MODE_SGBM);
        else if (alg == STEREO_HH4)
            sgbm->setMode(StereoSGBM::MODE_HH4);
        else if (alg == STEREO_3WAY)
            sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

        Mat disp, disp8;
        float multiplier = 1.0f;
        int64 t = getTickCount();

        if (alg == STEREO_BM) {
            bm->compute(img1, img2, disp);
            multiplier = 16.0f;
        }
        else {
            sgbm->compute(img1, img2, disp);
            multiplier = 16.0f;
        }

        t = getTickCount() - t;
        std::cout << "Elapsed time: " << t * 1000 / getTickFrequency() << "ms\n";

        disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities * multiplier));
        Mat disp_color;
        if (color_display)
            applyColorMap(disp8, disp_color, COLORMAP_TURBO);

        if (!disparity_filename.empty()) {
            std::ostringstream oss;
            oss << disparity_filename << "_" << pair_idx << ".png";
            imwrite(oss.str(), color_display ? disp_color : disp8);
        }

        if (!point_cloud_filename.empty() && !Q.empty()) {
            Mat xyz, float_disp;
            disp.convertTo(float_disp, CV_32F, 1.0f / multiplier);
            reprojectImageTo3D(float_disp, xyz, Q, true);
            std::ostringstream oss;
            oss << point_cloud_filename << "_" << pair_idx << ".xyz";
            saveXYZ(oss.str().c_str(), xyz);
        }

        if (!no_display) {
            imshow("left", img1);
            imshow("right", img2);
            imshow("disparity", color_display ? disp_color : disp8);
            std::cout << "Press ESC to continue to next pair..." << std::endl;
            if (waitKey(0) == 27) continue;
        }
    }

    return 0;
}

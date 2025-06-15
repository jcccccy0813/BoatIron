/*
 * stereo_match.cpp
 * calibration
 *
 * Modified to output XYZ files with embedded color information (X Y Z R G B)
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <sstream>
#include <iostream>
#include <fstream>
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

static void saveColoredXYZ(const char* filename, const Mat& mat, const Mat& color_img)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    if (!fp) {
        cerr << "Failed to open " << filename << " for writing" << endl;
        return;
    }

    // Convert color image to RGB format if grayscale
    Mat color_rgb;
    if (color_img.channels() == 1) {
        cvtColor(color_img, color_rgb, COLOR_GRAY2RGB);
    }
    else {
        color_rgb = color_img.clone();
    }

    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            Vec3f point = mat.at<Vec3f>(y, x);
            if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)continue;              
            Vec3b color = color_rgb.at<Vec3b>(y, x);
            fprintf(fp, "%f %f %f %d %d %d\n",
                point[0], point[1], point[2],
                color[2], color[1], color[0]); // OpenCV is BGR order, convert to RGB
        }
    }
    fclose(fp);
    cout << "Saved colored point cloud to " << filename << endl;
}

int main1(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{help h||}{list||}{algorithm|sgbm|}{max-disparity|64|}{blocksize|5|}"
        "{no-display||}{color||}{scale|1|}{i||}{e||}{o||}{p||}");

    if (parser.has("help")) {
        print_help(argv);
        return 0;
    }

    string list_file = parser.get<string>("list");
    string intrinsic_filename = parser.get<string>("i");
    string extrinsic_filename = parser.get<string>("e");
    string disparity_filename = parser.get<string>("o");
    string point_cloud_filename = parser.get<string>("p");
    string algorithm = parser.get<string>("algorithm");

    int numberOfDisparities = parser.get<int>("max-disparity");
    int SADWindowSize = parser.get<int>("blocksize");
    float scale = parser.get<float>("scale");
    bool no_display = parser.has("no-display");
    bool color_display = parser.has("color");

    if (list_file.empty()) {
        cerr << "Error: Please provide --list=<image_list.txt>" << endl;
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
        cerr << "Unknown algorithm: " << algorithm << endl;
        return -1;
    }

    Ptr<StereoBM> bm = StereoBM::create(16, 9);
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

    ifstream infile(list_file);
    if (!infile.is_open()) {
        cerr << "Failed to open list file: " << list_file << endl;
        return -1;
    }

    string left_path, right_path;
    int pair_idx = 0;
    while (infile >> left_path >> right_path) {
        cout << "\n[INFO] Processing pair #" << ++pair_idx << ": " << left_path << " " << right_path << endl;

        Mat img1 = imread(left_path, alg == STEREO_BM ? IMREAD_GRAYSCALE : IMREAD_COLOR);
        Mat img2 = imread(right_path, alg == STEREO_BM ? IMREAD_GRAYSCALE : IMREAD_COLOR);
        if (img1.empty() || img2.empty()) {
            cerr << "Could not load image pair: " << left_path << ", " << right_path << endl;
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
                cerr << "Failed to open intrinsic file." << endl;
                return -1;
            }
            Mat M1, D1, M2, D2;
            fs["M1"] >> M1; fs["D1"] >> D1;
            fs["M2"] >> M2; fs["D2"] >> D2;
            M1 *= scale; M2 *= scale;

            fs.open(extrinsic_filename, FileStorage::READ);
            if (!fs.isOpened()) {
                cerr << "Failed to open extrinsic file." << endl;
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
        //生成矫正后图像
        imwrite("1.jpg",img1);
        imwrite("2.jpg", img2);
        if (alg == STEREO_BM) {
            bm->compute(img1, img2, disp);
            multiplier = 16.0f;
        }
        else {
            sgbm->compute(img1, img2, disp);
            multiplier = 16.0f;
        }

        t = getTickCount() - t;
        cout << "Elapsed time: " << t * 1000 / getTickFrequency() << "ms\n";

        disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities * multiplier));
        Mat disp_color;
        if (color_display)
            applyColorMap(disp8, disp_color, COLORMAP_TURBO);

        if (!disparity_filename.empty()) {
            ostringstream oss;
            oss << disparity_filename << "_" << pair_idx << ".png";
            imwrite(oss.str(), color_display ? disp_color : disp8);
        }

        if (!point_cloud_filename.empty() && !Q.empty()) {
            Mat xyz, float_disp;
            disp.convertTo(float_disp, CV_32F, 1.0f / multiplier);
            reprojectImageTo3D(float_disp, xyz, Q, true);


            ostringstream oss;
            oss << point_cloud_filename << "_" << pair_idx << ".xyz";

            // Use original color image or grayscale image as color source
            Mat color_source = (img1.channels() == 3) ? img1 : disp8;
            saveColoredXYZ(oss.str().c_str(), xyz, color_source);
        }

        if (!no_display) {
            imshow("left", img1);
            imshow("right", img2);
            imshow("disparity", color_display ? disp_color : disp8);
            cout << "Press ESC to continue to next pair..." << endl;
            if (waitKey(0) == 27) continue;
        }
    }

    return 0;
}
//
//  lib.h
//  pi
//
//  Created by Felipe Teles on 15/04/24.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

enum SobelDirection {
    x, // Valor 0
    y, // Valor 1
    xy, // Valor 2
};

vector<int> calculate_histogram(const Mat &image, int max_lvl);
Mat quantize_image(Mat img,int maxLevel, int level);
Mat draw_histogram(const vector<int>& histogram, int max_lvl, int height);
vector<int> caculate_cumulative_histogram(const Mat &image, int max_lvl);
Mat draw_histogram(const vector<int>& histogram, int max_lvl, int height);
Mat match_histograms(const Mat &target, const vector<int> &source_cumalative_histogram, int max_lvl);
Mat gauss_kernel(int kernel_size_x, int kernel_size_y, float sigma);
Mat apply_kernel(Mat img, Mat kernel);
Mat otsu_threshold(Mat img);
Mat apply_blur(Mat image);
Mat get_canny(Mat img_blur);
Mat get_sobel(Mat img_blur, SobelDirection dir = xy);
Mat get_laplacian(Mat img);
Mat get_gradient(Mat img);
Mat get_prewitt(Mat img);
Mat apply_kmeans(Mat img);

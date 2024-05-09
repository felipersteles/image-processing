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

vector<int> calculate_histogram(const Mat &image, int max_lvl);
Mat quantize_image(Mat img,int maxLevel, int level);
Mat draw_histogram(const vector<int>& histogram, int max_lvl, int height);
vector<int> caculate_cumulative_histogram(const Mat &image, int max_lvl);
Mat draw_histogram(const vector<int>& histogram, int max_lvl, int height);
Mat match_histograms(const Mat &target, const vector<int> &source_cumalative_histogram, int max_lvl);
Mat gauss_kernel(int kernel_size_x, int kernel_size_y, float sigma);
Mat apply_kernel(Mat img, Mat kernel);
Mat otsu_threshold(Mat img);

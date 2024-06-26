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

vector<int> calculate_histogram(const Mat &image, int max_lvl = 256);
Mat quantize_image(Mat img,int maxLevel, int level);
void draw_histogram(const vector<int>& histogram, string title, int max_level = 256, int height = 500);
vector<int> caculate_cumulative_histogram(const Mat &image, int max_lvl);
Mat draw_histogram(const vector<int>& histogram, int max_lvl, int height);
Mat equalize_hist(Mat img);
Mat clahe(Mat img, int tileSize = 128, int clipLimit = 16);
Mat match_histograms(const Mat &target, const vector<int> &source_cumalative_histogram, int max_lvl);
Mat gauss_kernel(int kernel_size_x, int kernel_size_y, float sigma);
Mat apply_kernel(Mat img, Mat kernel);
Mat otsu_threshold(Mat img);
Mat apply_blur(Mat image);
Mat unsharp(Mat image, float amount = 1);
Mat get_canny(Mat img, double threshold1 = 100, double threshold2 = 200);
Mat get_sobel(Mat img_blur, SobelDirection dir = xy);
Mat get_laplacian(Mat img);
Mat get_gradient(Mat img);
Mat get_prewitt(Mat img);
Mat apply_kmeans(Mat img);
bool verify_neighbors(const Mat& image, int row, int col);
Mat create_circular_kernel(int radius);
cv::Mat vessel_enhancement(const cv::Mat& image, double sigma1 = 5.0, double sigma2 = 1.0);
Mat apply_superpixel(Mat img,  int region_size = 500, float regularity = 2, int num_iterations = 2);
Mat trasnform_image(const cv::Mat& image);
cv::Mat multi_scale_line_enhancement(const cv::Mat& image, double sigmaMin = 1, double sigmaMax = 2);
Mat erode(const Mat& image, Mat& kernel, int iterations = 1);
Mat dilate(const Mat& image, Mat& kernel, int iterations = 1);
Mat closing(const Mat& image, Mat& kernel, int iterations = 1);
cv::Mat opening(const cv::Mat& image, const cv::Mat& kernel, int iterations);
Mat extract_SIFT_features(const cv::Mat& image);
Mat extract_SURF_features(const cv::Mat& image);
cv::Mat calculate_bayer_probs(const cv::Mat& image, const cv::Mat& mask,
                              double foreground_prob_threshold = 0.1,
                              double background_prob_threshold = 0.1,
                              double default_foreground_prob = 0.9);
Mat region_growing(const Mat& anImage,
                  const pair<int, int>& aSeedSet,
                  unsigned char anInValue = 255,
                  float tolerance = 5);
Mat select_area(const Mat& binary_image, const Point& seed_point);

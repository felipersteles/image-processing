//
//  utils.hpp
//  pi
//
//  Created by Felipe Teles on 02/07/24.
//

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <dirent.h>  // For directory handling
#include <cmath>  // for mathematical functions

using namespace std;
using namespace cv;

std::vector<std::pair<cv::Mat, cv::Mat>> load_mask_atlas(std::vector<cv::Mat> images, std::vector<cv::Mat> masks);
std::vector<std::pair<cv::Mat, cv::Mat>> load_features_atlas(std::vector<cv::Mat> images);
std::vector<std::pair<cv::Mat, cv::Mat>> load_bayers_atlas(std::vector<cv::Mat> images, std::vector<cv::Mat> masks);
std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> load_all_atlas(std::vector<cv::Mat> images, std::vector<cv::Mat> masks);
std::vector<string> loadImagesNames(const std::string& folder_path);
std::vector<cv::Mat> loadImages(const std::string& folder_path, string name);
void cut_image(cv::Mat image, const cv::Mat atlas);


float calculate_specificity(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask);
float calculate_dice_coefficient(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask);
float calculate_jaccard_index(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask);
float calculate_recall(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask);
float calculate_precision(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask);
float calculate_accuracy(const Mat& segmented_image, const Mat& real_mask);

#endif /* utils_hpp */

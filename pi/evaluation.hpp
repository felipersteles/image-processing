//
//  evaluation.hpp
//  pi
//
//  Created by Felipe Teles on 17/06/24.
//

#ifndef evaluation_hpp
#define evaluation_hpp

#include <stdio.h>
#include <cmath>  // for mathematical functions
#include <opencv2/opencv.hpp>

float calculate_specificity(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask);
float calculate_dice_coefficient(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask);
float calculate_jaccard_index(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask);
void evaluate_segmentation();

#endif /* evaluation_hpp */

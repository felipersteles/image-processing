//
//  pancreas_segmentation.hpp
//  pi
//
//  Created by Felipe Teles on 06/06/24.
//

#ifndef pancreas_segmentation_hpp
#define pancreas_segmentation_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <dirent.h>  // For directory handling
#include <cmath>  // for mathematical functions

cv::Mat get_atlas(const std::vector<cv::Mat>& list_images);
cv::Mat reduce_mask(const cv::Mat& mask);
cv::Mat pre_process_img(const cv::Mat& img);
cv::Mat pos_process_img(const cv::Mat& img);
void see_image(std::string image_name);

cv::Mat map_segmentation(cv::Mat& img, std::vector<std::pair<Mat, Mat>> mask_atlas, cv::Mat atlas);


#endif /* pancreas_segmentation_hpp */

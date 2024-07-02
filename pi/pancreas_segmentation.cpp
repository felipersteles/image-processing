//
//  pancreas_segmentation.cpp
//  pi - digital image processing
//
//  Created by Felipe Teles on 06/06/24.
//

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include <dirent.h>  // For directory handling
#include <cmath>  // for mathematical functions

#include "lib.h"
#include "utils.hpp"
#include "pancreas_segmentation.hpp"

using namespace std;
using namespace cv;


cv::Mat reduce_mask(const cv::Mat& mask){
    
    Mat image = mask.clone();
    
    //    // Here, creating a 3x3 rectangular kernel
    Mat kernel = Mat::ones(5, 5, CV_8U);
    //
    //  Perform closing
    Mat result = erode(image, kernel, 3);
    
    return result;
}

cv::Mat pre_process_img(const cv::Mat& img){
    
    Mat image = img.clone();
    
    // Preprocess image (e.g., normalization)
    cv::Mat image_equalized = equalize_hist(image);
    Mat image_unsharp = unsharp(image_equalized);
    
    cv::Mat binary = otsu_threshold(image_unsharp);
    //
    //    // Define the kernel (structuring element)
    //    // Here, creating a 3x3 rectangular kernel
    Mat kernel = Mat::ones(3, 3, CV_8U);
    //
    //  Perform closing
    Mat result = closing(binary, kernel, 2);
    
    return image_unsharp;
}

cv::Mat pos_process_img(const cv::Mat& img){
    
    Mat image = img.clone();
    
    // Preprocess image (e.g., normalization)
    cv::Mat image_equalized = equalize_hist(image);
    Mat blur_img = apply_blur(image_equalized);
    Mat image_unsharp = unsharp(blur_img);
    
    Mat superpixel = apply_superpixel(image_unsharp, 50, 2, 8);
//        Mat blur_img = apply_blur(superpixel);
    
    cv::Mat binary = otsu_threshold(superpixel);
    
        Mat op_kernel = Mat::ones(3, 3, CV_8U);
    Mat cl_kernel = Mat::ones(2, 2, CV_8U);
        Mat closed = closing(binary, op_kernel, 1);
    Mat new_img = erode(closed, cl_kernel, 5);
    
    //
    //    // Define the kernel (structuring element)
    //    // Here, creating a 3x3 rectangular kernel
    Mat closing_kernel = Mat::ones(2,3, CV_8U);

    
    //  Perform closing
//    Mat opened = opening(binary, closing_kernel, 3);
//    Mat new_img = erode(opened, opening_kernel, 1);
    Mat closeda = dilate(new_img, closing_kernel, 2);
    //    Mat result = opening(closed, opening_kernel, 1);
    
    return closeda;
}

double feature_similarity(const cv::Mat& descriptors1, const cv::Mat& descriptors2) {
    
    // Matching keypoints using FlannBasedMatcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2 );
    
    // Filter good matches based on Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    
    // Calculate average distance between good matches
    double total_distance = 0.0;
    if (!good_matches.empty()) {
        for (const auto& m : good_matches) {
            total_distance += m.distance;
        }
        return total_distance / good_matches.size(); // Average distance
    }
    
    // Handle no good matches (optional: return high distance or -1)
    return std::numeric_limits<double>::max();  // Or return -1 to indicate very dissimilar
}

double find_location(const cv::Mat& image, const cv::Mat& descriptor, const cv::Mat& ref_mask) {
    // Calculate NCC for reference mask similarity (consider pre-computing reference NCCs for efficiency)
    cv::Mat ref_ncc;
    cv::matchTemplate(image, ref_mask, ref_ncc, cv::TM_CCOEFF_NORMED); // Normalized cross-correlation
    double ref_similarity = cv::mean(ref_ncc)[0]; // Average NCC value across the mask
    
    // Feature detection and description
    cv::Mat img_descriptor = extract_SIFT_features(image);
    
    // Calculate NCC for feature similarity with each reference descriptor
    double best_feature_similarity = -1.0; // Normalized values range from -1 to 1 (perfect match)
    int best_match_idx = -1;
    
    cv::Mat feature_ncc;
    cv::matchTemplate(img_descriptor, descriptor, feature_ncc, cv::TM_CCOEFF_NORMED);
    double current_similarity = cv::mean(feature_ncc)[0];
    
    // You can return the best feature similarity or a combination of reference and feature similarity
    // based on your application's needs. Here's an example combining them:
    double combined_score = ref_similarity * best_feature_similarity;
    
    return combined_score;
}

cv::Mat map_segmentation(cv::Mat& img,
                         std::vector<std::pair<Mat, Mat>> mask_atlas) {
    
    Mat image_preprocessed = pre_process_img(img);
    Mat image_descriptor = extract_SIFT_features(image_preprocessed);
    
    // Find the most similar image-mask pair in the atlas based on image similarity
    double min_distance = std::numeric_limits<double>::max();
    int best_idx = -1;
    
    // Loop through each image-mask pair in the atlas
    for (size_t i = 0; i < mask_atlas.size(); ++i) {
        const cv::Mat& mask = mask_atlas[i].second;
        
        // Calculate image similarity metric (consider alternatives like normalized cross-correlation)
        double distance = cv::norm(image_preprocessed, mask, cv::NORM_L1); // L1 norm for simplicity
        
        // Heuristic 1: Check similarity and minimum intensity difference (consider adaptive thresholding)
        double intensity_diff = std::abs(cv::mean(image_preprocessed)[0] - cv::mean(mask)[0]);
        double intensity_threshold = 10; // Adjust based on your data
        if (distance < min_distance && intensity_diff > intensity_threshold) {
            min_distance = distance;
            best_idx = (int) i;
        }
    }
    
    // Output segmentation mask
    cv::Mat segmentation_mask = cv::Mat::zeros(image_preprocessed.size(), CV_8UC1);
    
    // If a similar image-mask pair is found, use its mask with refinements
    if (best_idx != -1) {
        cv::Mat original_mask = mask_atlas[best_idx].second.clone();  // Clone to avoid modifying original atlas data
        cv::Mat mask = reduce_mask(original_mask);
        
        Mat pos_processed_img = pos_process_img(img);
        for (int y = 0; y < pos_processed_img.rows; ++y) {
            for (int x = 0; x < pos_processed_img.cols; ++x) {
                if(mask.at<uchar>(y,x) > 0
                   && pos_processed_img.at<uchar>(y,x) > 0
                   && verify_neighbors(pos_processed_img, y, x)
                   ){
                    std::cout << "| [Info]: Pancreas found on pixel(" << x <<", "<< y <<")." << std::endl;
                    return select_area(pos_processed_img, Point(x,y));
                }
            }
        }
    } else {
        // Handle the entire image without a good match (consider different background handling strategies)
        segmentation_mask = cv::Scalar(0); // Assign background value (optional: explore background modeling)
    }
    
    
    return segmentation_mask;
}

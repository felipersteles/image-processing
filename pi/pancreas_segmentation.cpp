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
#include <limits>

#include "lib.h"
#include "utils.hpp"
#include "pancreas_segmentation.hpp"

using namespace std;
using namespace cv;

cv::Mat get_atlas(const std::vector<cv::Mat>& list_images) {
    // Ensure all images are grayscale
    for (const cv::Mat& image : list_images) {
        if (image.channels() != 1) {
            std::cerr << "Error: Non-grayscale image found. Only grayscale images allowed." << std::endl;
            return cv::Mat(); // Return empty Mat on error (optional)
        }
    }
    
    cv::Mat atlas;
    atlas = cv::Mat::zeros(list_images[0].size(), list_images[0].type());
    
    for (size_t i = 0; i < list_images.size(); ++i) {
        std::cout << "| [Info - Atlas]: Processing Image " << i + 1 << " of " << list_images.size() << std::endl;
        
        // Check if image is valid
        if (list_images[i].empty()) {
            std::cerr << "Error: Invalid image at index " << i << std::endl;
            continue; // Skip invalid image
        }
        
        Mat image = list_images[i].clone();
        image /= 255;
        atlas += image;
    }
    
    return atlas;
}


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
    
    return image_unsharp;
}

cv::Mat pos_process_img(const cv::Mat& img){
    
    Mat image = img.clone();
    
    // Preprocess image (e.g., normalization)
    cv::Mat image_equalized = equalize_hist(image);
    Mat blur_img = apply_blur(image_equalized);
    Mat image_unsharp = unsharp(blur_img);
    
    // dividir a região dos orgãos
    Mat superpixel = apply_superpixel(image_unsharp, 50, 2, 8);
    
    // binarização
    cv::Mat binary = otsu_threshold(superpixel);
    
    
    Mat cl_kernel = Mat::ones(3, 3, CV_8U);
    Mat closed = closing(binary, cl_kernel, 1);
    
    Mat er_kernel = Mat::ones(2, 2, CV_8U);
    Mat new_img = erode(closed, er_kernel, 5);
    
    Mat closing_kernel = Mat::ones(2,3, CV_8U);
    Mat result = dilate(new_img, closing_kernel, 2);
    
    return result;
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

bool verify_mask(const cv::Mat& mask, const cv::Mat& atlas, int threshold = 200,
                 int best_x_offset = 0, int best_y_offset = 0) {
    // Check data type compatibility (assuming CV_8UC1 for mask)
    if (mask.type() != atlas.type()) {
        std::cerr << "Error: Incompatible data types for mask and atlas." << std::endl;
        return false;
    }
    
    const int mask_rows = mask.rows;
    const int mask_cols = mask.cols;
    const int atlas_rows = atlas.rows;
    const int atlas_cols = atlas.cols;
    
    // Maximum allowed offset to prevent out-of-bounds access
    const int max_offset = std::min(atlas_rows - mask_rows, atlas_cols - mask_cols);
    
    // Initialize variables for best fit
    float best_fit_score = -std::numeric_limits<float>::infinity();
    best_x_offset = 0;
    best_y_offset = 0;
    
    // Loop through possible offsets
    for (int y_offset = 0; y_offset <= max_offset; ++y_offset) {
        for (int x_offset = 0; x_offset <= max_offset; ++x_offset) {
            float fit_score = 0.0f;
            
            // Check overlapping area of mask and atlas
            for (int y = 0; y < mask_rows; ++y) {
                for (int x = 0; x < mask_cols; ++x) {
                    int atlas_y = y + y_offset;
                    int atlas_x = x + x_offset;
                    
                    // Check if within atlas bounds
                    if (atlas_y >= 0 && atlas_y < atlas_rows &&
                        atlas_x >= 0 && atlas_x < atlas_cols) {
                        uchar mask_intensity = mask.at<uchar>(y, x);
                        if (mask_intensity == 255) {
                            float atlas_value = atlas.at<uchar>(atlas_y, atlas_x);
                            if (atlas_value > threshold) {
                                fit_score += atlas_value; // Accumulate score for high-intensity matches
                            }
                        }
                    }
                }
            }
            
            // Update best fit if current score is higher
            if (fit_score > best_fit_score) {
                best_fit_score = fit_score;
                best_x_offset = x_offset;
                best_y_offset = y_offset;
            }
        }
    }
    
    return best_fit_score > 5.0f; // Return true if a match is found
}

cv::Mat map_segmentation(cv::Mat& img, std::vector<std::pair<Mat, Mat>> mask_atlas, cv::Mat atlas) {
    
    Mat image_preprocessed = pre_process_img(img);
    
    // Find the most similar image-mask pair in the atlas based on image similarity
    double min_distance_mask = std::numeric_limits<double>::max();
    int best_idx = -1;
    
    // Loop through each image-mask pair in the atlas
    for (size_t i = 0; i < mask_atlas.size(); ++i) {
        const cv::Mat& mask = mask_atlas[i].second;
        vector<int> hist = calculate_histogram(mask_atlas[i].first);
        
        // Calculate image similarity metric (consider alternatives like normalized cross-correlation)
        double distance_mask = cv::norm(image_preprocessed, mask, cv::NORM_L1); // L1 norm for simplicity
        bool good_fit = verify_mask(mask, atlas);
        
        if (good_fit && distance_mask < min_distance_mask) {
            min_distance_mask = distance_mask;
            best_idx = (int) i;
        }
    }
    
    //     debug
    //    imshow("Mask found",mask_atlas[best_idx].second);
    
    // Output segmentation mask
    cv::Mat segmentation_mask = cv::Mat::zeros(image_preprocessed.size(), CV_8UC1);
    if (best_idx != -1) {
        cv::Mat original_mask = mask_atlas[best_idx].second.clone();  // Clone to avoid modifying original atlas data
        cv::Mat mask = reduce_mask(original_mask);
        
        Mat pos_processed_img = pos_process_img(img);
        cut_image(pos_processed_img, atlas);
        for (int y = 0; y < pos_processed_img.rows; ++y) {
            for (int x = 0; x < pos_processed_img.cols; ++x) {
                if(
                   mask.at<uchar>(y,x) > 0
                   && pos_processed_img.at<uchar>(y,x) > 0
                   && verify_neighbors(pos_processed_img, y, x)
                   ){
                       std::cout << "| [Info]: Pancreas found on pixel(" << x <<", "<< y <<")." << std::endl;
                       return select_area(pos_processed_img, Point(x,y));
                   }
            }
        }
    } else {
        segmentation_mask = cv::Scalar(0);
        std::cout << "| [Info]: Not found." << std::endl;
    }
    
    
    return segmentation_mask;
}

void see_image(std::string image_name){
    cv::Mat img = cv::imread("/Users/felipeteles/Development/ufma/dataset/test/img/" + image_name, IMREAD_GRAYSCALE);
    cv::Mat mask = cv::imread("/Users/felipeteles/Development/ufma/dataset/test/mask/" + image_name, IMREAD_GRAYSCALE);
    
    
    std::string train_mask_folder = "/Users/felipeteles/Development/ufma/dataset/train/mask/";
    std::vector<cv::Mat> train_masks = loadImages(train_mask_folder, "Train masks");
    cv:Mat prob_atlas = get_atlas(train_masks);
    
    
    cv::Mat pos_img = pos_process_img(img);
    cv::Mat use_mask = reduce_mask(mask);
    cut_image(pos_img, prob_atlas);
    
    cv::Mat output;
    bool find;
    for (int y = 0; y < pos_img.rows; ++y) {
        for (int x = 0; x < pos_img.cols; ++x) {
            if(use_mask.at<uchar>(y,x) > 0 && pos_img.at<uchar>(y,x) > 0 && verify_neighbors(pos_img, y, x)){
                std::cout << "| [Info]: Pancreas found on pixel(" << x <<", "<< y <<")." << std::endl;
                output = select_area(pos_img, Point(x, y));
                find = true;
                break;
            }
        }
        
        if(find) break;
    }
    
    imshow("teste", pos_img);
    waitKey(0);
}

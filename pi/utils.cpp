//
//  utils.cpp
//  pi
//
//  Created by Felipe Teles on 02/07/24.
//

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <dirent.h>  // For directory handling
#include <cmath>  // for mathematical functions

#include "utils.hpp"
#include "pancreas_segmentation.hpp"
#include "lib.h"

using namespace std;
using namespace cv;

// Function to load images from a folder
std::vector<cv::Mat> loadImages(const std::string& folder_path, string name) {
    
    std::vector<cv::Mat> images;
    DIR* dir;
    struct dirent* ent;
    std::string filename;
    
    // Open the directory
    if ((dir = opendir(folder_path.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            filename = ent->d_name;
            
            // Check if it's a regular file and has an image extension
            if (ent->d_type == DT_REG && (filename.find(".jpg") != std::string::npos ||
                                          filename.find(".png") != std::string::npos ||
                                          filename.find(".bmp") != std::string::npos)) {
                // Construct the full path
                std::string full_path = folder_path + "/" + filename;
                
                // Read the image using imread
                cv::Mat image = cv::imread(full_path, IMREAD_GRAYSCALE);
                
                // Check if image loaded successfully
                if (!image.empty()) {
                    images.push_back(image);
                } else {
                    std::cerr << "Error loading image: " << full_path << std::endl;
                }
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory: " << folder_path << std::endl;
    }
    
    
    std::cout << "| [Info]: " << name <<" images successfully loaded! " << std::endl;
    return images;
}

// Function to load images from a folder
std::vector<string> loadImagesNames(const std::string& folder_path) {
    
    std::vector<string> images;
    DIR* dir;
    struct dirent* ent;
    std::string filename;
    
    // Open the directory
    if ((dir = opendir(folder_path.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            filename = ent->d_name;
            
            // Check if it's a regular file and has an image extension
            if (ent->d_type == DT_REG && (filename.find(".jpg") != std::string::npos ||
                                          filename.find(".png") != std::string::npos ||
                                          filename.find(".bmp") != std::string::npos)) {
                images.push_back(filename);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory: " << folder_path << std::endl;
    }
    
    return images;
}

// Function to load probabilistic atlas (implementation depends on format)
std::vector<std::pair<cv::Mat, cv::Mat>> load_mask_atlas(std::vector<cv::Mat> images, std::vector<cv::Mat> masks) {
    
    // Check for successful loading
    if (images.empty()) {
        std::cerr << "| [Error]: No images found." << std::endl;
        return std::vector<std::pair<cv::Mat, cv::Mat>>();
    }
    
    if (masks.empty()) {
        std::cerr << "| [Error]: No masks found." << std::endl;
        return std::vector<std::pair<cv::Mat, cv::Mat>>();
    }
    
    // Assuming a correspondence between images and masks (e.g., matching filenames)
    if (images.size() != masks.size()) {
        std::cerr << "| [Warning]: Number of images and masks don't match." << std::endl;
    }
    
    // Create a vector to store the probabilistic atlas pairs (image, mask)
    std::vector<std::pair<cv::Mat, cv::Mat>> atlas;
    
    // Combine images and masks into the atlas
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat img = pre_process_img(images[i]);
        
        atlas.push_back(std::make_pair(img, masks[i]));
    }
    
    std::cout << "| [Info]: Mask atlas successfully loaded." << std::endl;
    return atlas;
}

// Function to load probabilistic atlas (implementation depends on format)
std::vector<std::pair<cv::Mat, cv::Mat>> load_features_atlas(std::vector<cv::Mat> images) {
    
    // Create a vector to store the probabilistic atlas pairs (image, mask)
    std::vector<std::pair<cv::Mat, cv::Mat>> atlas;
    
    // Combine images and masks into the atlas
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat img = pre_process_img(images[i]);
        cv::Mat descriptor = extract_SIFT_features(img);
        atlas.push_back(std::make_pair(img, descriptor));
    }
    
    std::cout << "| [Info]: Feature atlas successfully loaded." << std::endl;
    return atlas;
}

// Function to load probabilistic atlas (implementation depends on format)
std::vector<std::pair<cv::Mat, cv::Mat>> load_bayers_atlas(std::vector<cv::Mat> images, std::vector<cv::Mat> masks) {
    
    // Create a vector to store the probabilistic atlas pairs (image, mask)
    std::vector<std::pair<cv::Mat, cv::Mat>> atlas;
    
    // Combine images and masks into the atlas
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat prob = calculate_bayer_probs(images[i], masks[i]);
        
        atlas.push_back(std::make_pair(images[i], prob));
    }
    
    std::cout << "| [Info]: Bayers cartesian atlas successfully loaded." << std::endl;
    return atlas;
}

std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> load_all_atlas(std::vector<cv::Mat> images, std::vector<cv::Mat> masks){
    
    std::vector<std::pair<Mat, Mat>> mask_atlas;
    std::vector<std::pair<Mat, Mat>> features_atlas;
    //    std::vector<std::pair<Mat, Mat>> bayers_atlas;
    
    // Combine images and masks into the atlas
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat img = pre_process_img(images[i]);
        cv::Mat descriptor = extract_SIFT_features(img);
        //        cv::Mat prob = calculate_bayer_probs(img, masks[i]);
        
        mask_atlas.push_back(std::make_pair(img, masks[i]));
        features_atlas.push_back(std::make_pair(img, descriptor));
        //        bayers_atlas.push_back(std::make_pair(img, prob));
    }
    
    std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> all_atlas;
    all_atlas.push_back(mask_atlas);
    all_atlas.push_back(features_atlas);
    //    all_atlas.push_back(bayers_atlas);
    
    std::cout << "| [Info]: Mask atlas successfully loaded." << std::endl;
    std::cout << "| [Info]: Feature atlas successfully loaded." << std::endl;
    //    std::cout << "| [Info]: Bayers cartesian atlas successfully loaded." << std::endl;
    
    return all_atlas;
}

/**
 * Calculates the specificity metric for image segmentation.
 *
 * @param segmentation_mask The predicted segmentation mask (single-channel CV_8UC1 Mat).
 * @param ground_truth_mask The ground truth mask (single-channel CV_8UC1 Mat).
 *
 * @return The specificity value (float between 0 and 1).
 */
float calculate_specificity(const Mat& segmentation_mask, const Mat& ground_truth_mask) {
    // Assert same sizes for segmentation and ground truth masks
    CV_Assert(segmentation_mask.size() == ground_truth_mask.size());
    
    // Initialize variables for counting true negatives (TN) and total negatives (all background pixels in ground truth)
    int true_negatives = 0;
    int total_negatives = 0;
    
    // Iterate through each pixel
    for (int y = 0; y < segmentation_mask.rows; ++y) {
        for (int x = 0; x < segmentation_mask.cols; ++x) {
            uchar segmentation_value = segmentation_mask.at<uchar>(y, x);
            uchar ground_truth_value = ground_truth_mask.at<uchar>(y, x);
            
            // Count true negatives (background in both predicted and ground truth)
            if (segmentation_value == 0 && ground_truth_value == 0) {
                true_negatives++;
            }
            
            // Count total negatives (background pixels in ground truth)
            if (ground_truth_value == 0) {
                total_negatives++;
            }
        }
    }
    
    // Specificity calculation (TN / total negatives), handling division by zero
    float specificity = 0.0f;
    if (total_negatives > 0) {
        specificity = static_cast<float>(true_negatives) / total_negatives;
    } else {
        // If no true negatives, specificity is technically undefined but can be set to 1 (or any other value depending on the application)
        specificity = 1.0f; // Or another suitable value
    }
    
    return specificity;
}


/**
 * Calculates the Dice coefficient for image segmentation.
 *
 * @param segmentation_mask The predicted segmentation mask (single-channel CV_8UC1 Mat).
 * @param ground_truth_mask The ground truth mask (single-channel CV_8UC1 Mat).
 *
 * @return The Dice coefficient value (float between 0 and 1).
 */
float calculate_dice_coefficient(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask) {
    // Assert same sizes for segmentation and ground truth masks
    CV_Assert(segmentation_mask.size() == ground_truth_mask.size());
    
    // Initialize variables for counting true positives (TP), false positives (FP), and false negatives (FN)
    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    
    // Iterate through each pixel
    for (int y = 0; y < segmentation_mask.rows; ++y) {
        for (int x = 0; x < segmentation_mask.cols; ++x) {
            uchar segmentation_value = segmentation_mask.at<uchar>(y, x);
            uchar ground_truth_value = ground_truth_mask.at<uchar>(y, x);
            
            // Count true positives (foreground in both predicted and ground truth)
            if (segmentation_value > 0 && ground_truth_value > 0) {
                true_positives++;
            }
            
            // Count false positives (foreground predicted, background in ground truth)
            if (segmentation_value > 0 && ground_truth_value == 0) {
                false_positives++;
            }
            
            // Count false negatives (background predicted, foreground in ground truth)
            if (segmentation_value == 0 && ground_truth_value > 0) {
                false_negatives++;
            }
        }
    }
    
    // Dice coefficient calculation (2 * TP / (2 * TP + FP + FN))
    float dice_coefficient = 0.0f;
    int total_predicted_foreground = true_positives + false_positives;
    int total_ground_truth_foreground = true_positives + false_negatives;
    if (total_predicted_foreground > 0 && total_ground_truth_foreground > 0) {
        dice_coefficient = 2.0f * static_cast<float>(true_positives) / (total_predicted_foreground + total_ground_truth_foreground);
    }
    
    return dice_coefficient;
}

/**
 * Calculates the Jaccard Index for image segmentation.
 *
 * @param segmentation_mask The predicted segmentation mask (single-channel CV_8UC1 Mat).
 * @param ground_truth_mask The ground truth mask (single-channel CV_8UC1 Mat).
 *
 * @return The Jaccard Index value (float between 0 and 1).
 */
float calculate_jaccard_index(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask) {
    // Assert same sizes for segmentation and ground truth masks
    CV_Assert(segmentation_mask.size() == ground_truth_mask.size());
    
    // Initialize variables for counting true positives (TP), intersection (intersection of foreground pixels in both masks)
    int true_positives = 0;
    int intersection = 0;
    
    // Iterate through each pixel
    for (int y = 0; y < segmentation_mask.rows; ++y) {
        for (int x = 0; x < segmentation_mask.cols; ++x) {
            uchar segmentation_value = segmentation_mask.at<uchar>(y, x);
            uchar ground_truth_value = ground_truth_mask.at<uchar>(y, x);
            
            // Count true positives (foreground in both predicted and ground truth)
            if (segmentation_value > 0 && ground_truth_value > 0) {
                true_positives++;
                intersection++;
            }
        }
    }
    
    // Jaccard Index calculation (TP / (total predicted foreground + total ground truth foreground - intersection))
    float jaccard_index = 0.0f;
    int total_predicted_foreground = cv::countNonZero(segmentation_mask);
    int total_ground_truth_foreground = cv::countNonZero(ground_truth_mask);
    if (total_predicted_foreground + total_ground_truth_foreground > intersection) {
        jaccard_index = static_cast<float>(intersection) / (total_predicted_foreground + total_ground_truth_foreground - intersection);
    }
    
    return jaccard_index;
}

float calculate_recall(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask) {
    int true_positives = 0;
    int total_ground_truth_foreground = cv::countNonZero(ground_truth_mask);
    
    for (int y = 0; y < segmentation_mask.rows; ++y) {
        for (int x = 0; x < segmentation_mask.cols; ++x) {
            uchar segmentation_value = segmentation_mask.at<uchar>(y, x);
            uchar ground_truth_value = ground_truth_mask.at<uchar>(y, x);
            
            if (ground_truth_value > 0 && segmentation_value == ground_truth_value) {
                true_positives++;
            }
        }
    }
    
    // Recall calculation (true positives / total ground truth foreground pixels)
    float recall = static_cast<float>(true_positives) / total_ground_truth_foreground;
    return recall;
}

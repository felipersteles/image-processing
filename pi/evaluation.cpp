//
//  evaluation.cpp
//  pi
//
//  Created by Felipe Teles on 17/06/24.
//


#include <stdio.h>
#include <cmath>  // for mathematical functions
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#include "pancreas_segmentation.hpp"

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

void evaluate_segmentation(){
    
    
    // Define folder paths
    std::string train_image_folder = "/Users/felipeteles/Development/ufma/dataset/valid/img/";
    std::string train_mask_folder = "/Users/felipeteles/Development/ufma/dataset/valid/mask/";
    
    std::cout << "|------------------------------------------------------|" << std::endl;
    std::cout << "| Loading train images. " << std::endl;
    std::cout << "|------------------------------------------------------|" << std::endl;
    
    // Load images and masks
    std::vector<cv::Mat> train_images = loadImages(train_image_folder, "Train");
    std::vector<cv::Mat> train_masks = loadImages(train_mask_folder, "Train masks");
    
    // Ensure images and masks have the same size
    if (train_images.size() != train_masks.size()) {
        std::cerr << "| [Error]: Number of images and masks don't match!" << std::endl;
        return;
    }
    
    std::cout << "|------------------------------------------------------|" << std::endl;
    std::cout << "| Loading atlas. " << std::endl;
    std::cout << "|------------------------------------------------------|" << std::endl;
    
    //    std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> all_atlas = load_all_atlas(train_images, train_masks);
    
    //    std::vector<std::pair<Mat, Mat>> mask_atlas = all_atlas[0];
    //    std::vector<std::pair<Mat, Mat>> SIFT_atlas = all_atlas[1];
    
    std::vector<std::pair<Mat, Mat>> atlas = load_mask_atlas(train_images, train_masks);
    
    // Define folder paths
    std::string valid_image_folder = "/Users/felipeteles/Development/ufma/dataset/test/img/";
    std::string valid_mask_folder = "/Users/felipeteles/Development/ufma/dataset/test/mask/";
    
    std::cout << "|------------------------------------------------------|" << std::endl;
    std::cout << "| Loading valid images. " << std::endl;
    std::cout << "|------------------------------------------------------|" << std::endl;
    
    // Load images and masks
    std::vector<cv::Mat> valid_images = loadImages(valid_image_folder, "Valid");
    std::vector<cv::Mat> valid_masks = loadImages(valid_mask_folder, "Valid masks");
    std::vector<string> names = loadImagesNames(valid_image_folder);
    
    // Metrics
    float jaccard_index = 0.0f, dice_coefficient = 0.0f, specificity = 0.0f, recall = 0.0f;
//    std::vector<string> best_results;
    
    std::cout << "|------------------------------------------------------|" << std::endl;
    std::cout << "| Starting the pancreas automatic segmentation. " << std::endl;
    std::cout << "|------------------------------------------------------|" << std::endl;
    
    // Loop through each image-mask pair
    int img_count = 1;
    for (size_t i = 0; i < valid_images.size(); ++i) {
        cv::Mat image = valid_images[i];
        cv::Mat mask = valid_masks[i];
        
        
        cv::Mat output = map_segmentation(image, atlas);
        
        // Calculate segmentation metrics here (e.g., Jaccard Index, Dice Coefficient, Specificity)
        float jaccard_index_pair = calculate_jaccard_index(output, mask);
        float dice_coefficient_pair = calculate_dice_coefficient(output, mask);
        float specificity_pair = calculate_specificity(output, mask); // Assuming you have a function for specificity
        float  recall_pair = calculate_recall(output, mask);
        
        // Update overall metrics (consider averaging across all image-mask pairs)
        jaccard_index += jaccard_index_pair;
        dice_coefficient += dice_coefficient_pair;
        specificity += specificity_pair;
        recall += recall_pair;
        
        // Print the report of metrics
        std::cout << "| Image name: " << names[i] << std::endl;
        std::cout << "| Image index: " << img_count << std::endl;
        std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
        std::cout << "| Jaccard Index: " << jaccard_index_pair << std::endl;
        std::cout << "| Dice Coefficient: " << dice_coefficient_pair << std::endl;
        std::cout << "| Specificity: " << specificity_pair << std::endl;
        std::cout << "| Recall: " << recall_pair << std::endl;
        std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
        std::cout << "| Avarage metrics       |" << std::endl;
        std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
        std::cout << "| Jaccard Index: " << jaccard_index / img_count << std::endl;
        std::cout << "| Dice Coefficient: " << dice_coefficient / img_count << std::endl;
        std::cout << "| Specificity: " << specificity / img_count << std::endl;
        std::cout << "| Recall: " << recall / img_count << std::endl;
        std::cout << "|------------------------------------------------------|" << std::endl;
        std::cout << "| Missing: "<< valid_images.size() - img_count << std::endl;
        std::cout << "|------------------------------------------------------|" << std::endl;
        
        img_count++;
    }
    
    
    // Calculate average metrics if iterating through multiple image-mask pairs
    if (valid_images.size() > 1) {
        jaccard_index /= valid_images.size();
        dice_coefficient /= valid_images.size();
        specificity /= valid_images.size();
        recall /= valid_images.size();
    }
    
    // Print or store the calculated metrics (e.g., Jaccard Index, Dice Coefficient, Specificity)
    std::cout << "|----------------------------------------|" << std::endl;
    std::cout << "|-----------------------------------|" << std::endl;
    std::cout << "|------------------------------|" << std::endl;
    std::cout << "| Final results " << std::endl;
    std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
    std::cout << "| Jaccard Index: " << jaccard_index << std::endl;
    std::cout << "| Dice Coefficient: " << dice_coefficient << std::endl;
    std::cout << "| Specificity: " << specificity << std::endl;
    std::cout << "| Recall: " << recall << std::endl;
    std::cout << "|------------------------------|" << std::endl;
    std::cout << "|-----------------------------------|" << std::endl;
    std::cout << "|-------------------------Thanks ;)-----------|" << std::endl;
    std::cout << "|------------------------------------------------------|" << std::endl;
}

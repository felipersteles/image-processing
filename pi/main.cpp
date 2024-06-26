//
//  main.cpp
//  pi
//
//  Created by Felipe Teles on 15/04/24.
//

#include <opencv2/opencv.hpp>
#include <chrono>

#include "lib.h"
#include "evaluation.hpp"
#include "pancreas_segmentation.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    
    // perfome the segmentation then evaluate
        evaluate_segmentation();
    
    // Define folder paths
    //    std::string train_image_folder = "/Users/felipeteles/Development/ufma/dataset/valid/img/";
    //    std::string train_mask_folder = "/Users/felipeteles/Development/ufma/dataset/valid/mask/";
    
    //     Load images and masks
    //    std::vector<cv::Mat> train_images = loadImages(train_image_folder);
    //    std::vector<cv::Mat> train_masks = loadMasks(train_mask_folder);
    //
    //
    //    // load atlas
    //    std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> all_atlas = load_all_atlas(train_images, train_masks);
    ////
    //    std::vector<std::pair<Mat, Mat>> mask_atlas = all_atlas[0];
    //    std::vector<std::pair<Mat, Mat>> SIFT_atlas = all_atlas[1];
    ////    std::vector<std::pair<Mat, Mat>> HOG_atlas = all_atlas[2];
    //
    ////
//    ////     test pre processment
//    cv::Mat img = cv::imread("/Users/felipeteles/Development/ufma/dataset/test/img/pancreas_087_57.png", IMREAD_GRAYSCALE);
//    cv::Mat mask = cv::imread("/Users/felipeteles/Development/ufma/dataset/test/mask/pancreas_087_57.png", IMREAD_GRAYSCALE);
    
    //////
//        cv::Mat output = map_segmentation(img, mask_atlas, SIFT_atlas);
    //
    ////    Mat descriptor = extract_HOG_features(img);
    //
    //    std::cout << "SIFT descriptor rows: " << descriptor.rows << std::endl;
    //    std::cout << "HOG descriptor rows: " << hogDescriptors2.rows << std::endl;
    
//    cv::Mat new_img = pre_process_img(img);
//    cv::Mat other_img = pos_process_img(img);
//    cv::Mat use_mask = reduce_mask(mask);
////    //
//    cv::Mat output;
//    bool find;
//    for (int y = 0; y < other_img.rows; ++y) {
//        for (int x = 0; x < other_img.cols; ++x) {
//            if(use_mask.at<uchar>(y,x) > 0 && use_mask.at<uchar>(y,x) > 0 && verify_neighbors(other_img, y, x)){
//                std::cout << "| [Info]: Pancreas found on pixel(" << x <<", "<< y <<")." << std::endl;
//                output = select_area(other_img, Point(x, y));
//                find = true;
//                break;
//            }
//        }
//        if(find) break;
//    }
    //////
//    imshow("Image", img);
////    imshow("Pre processamento", new_img);
////    imshow("Mask", mask);
//    imshow("Mask reduzida", use_mask);
//    imshow("Pos processamento", other_img);
//    imshow("Segmentation", output);
    //////
    //////
//    waitKey(0);
    
    return 0;
}

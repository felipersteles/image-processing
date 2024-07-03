//
//  evaluation.hpp
//  pi
//
//  Created by Felipe Teles on 17/06/24.
//

#ifndef evaluation_hpp
#define evaluation_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <dirent.h>  // For directory handling
#include <cmath>  // for mathematical functions

#include "pancreas_segmentation.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

class Result{
public:
    int id;
    string name;
    Mat img;
    float dice;
    float jaccard_index;
    float specificity;
    float recall;
    
    Result(){};
    
    Result(
           int id,
           string name,
           Mat img,
           float dice,
           float jaccard_index,
           float specificity,
           float recall
           ){
        this->id = id;
        
        this->name = name;
        this-> img = img;
        
        this-> dice = dice;
        this-> jaccard_index = jaccard_index;
        this-> specificity = specificity;
        this-> recall = recall;
    };
    
    Mat get(){
        
        std::cout << std::endl;
        std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
        std::cout << "| Image name: " << this->name << std::endl;
        std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
        std::cout << "| Jaccard Index: " << this->jaccard_index << std::endl;
        std::cout << "| Dice Coefficient: " << this->dice << std::endl;
        std::cout << "| Specificity: " << this->specificity << std::endl;
        std::cout << "| Recall: " << this->recall << std::endl;
        std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
        std::cout << std::endl;
        
        return this->img;
    }
};

class Model{
public:
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> masks;
    std::vector<std::string> names;
    
    std::vector<Result> results;
    
    float dice_coefficient;
    float jaccard_index;
    float specificity;
    float recall;
    
    Model(){
        
        std::cout << "|------------------------------------------------------|" << std::endl;
        std::cout << "| Starting Pancreas Segmentation Model. " << std::endl;
        
        // Define folder paths
        std::string train_image_folder = "/Users/felipeteles/Development/ufma/dataset/train/img/";
        std::string train_mask_folder = "/Users/felipeteles/Development/ufma/dataset/train/mask/";
        
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
        
        std::vector<std::pair<Mat, Mat>> mask_atlas = load_mask_atlas(train_images, train_masks);
//        std::vector<std::pair<Mat, Mat>> feature_atlas = load_features_atlas(train_images);
        
        // Define folder paths
        std::string valid_image_folder = "/Users/felipeteles/Development/ufma/dataset/test/img/";
        std::string valid_mask_folder = "/Users/felipeteles/Development/ufma/dataset/test/mask/";
        
        std::cout << "|------------------------------------------------------|" << std::endl;
        std::cout << "| Loading valid images. " << std::endl;
        std::cout << "|------------------------------------------------------|" << std::endl;
        
        // Load images and masks
        this->images = loadImages(valid_image_folder, "Valid");
        this->masks = loadImages(valid_mask_folder, "Valid masks");
        this->names = loadImagesNames(valid_image_folder);
        
        std::cout << "|------------------------------------------------------|" << std::endl;
        std::cout << "| Starting the pancreas automatic segmentation. " << std::endl;
        std::cout << "|------------------------------------------------------|" << std::endl;
        
        
        this->jaccard_index = 0.0f;
        this->dice_coefficient = 0.0f;
        this->specificity = 0.0f;
        this->recall = 0.0f;
        
        // Loop through each image-mask pair
        int img_count = 0;
        for (size_t i = 0; i < this->images.size(); ++i) {
            cv::Mat image = this->images[i];
            cv::Mat mask = this->masks[i];
            string name = this->names[i];
            
            cv::Mat output = map_segmentation(image, mask_atlas);
            
            // Calculate segmentation metrics here (e.g., Jaccard Index, Dice Coefficient, Specificity)
            float jaccard_index_pair = calculate_jaccard_index(output, mask);
            float dice_coefficient_pair = calculate_dice_coefficient(output, mask);
            float specificity_pair = calculate_specificity(output, mask); // Assuming you have a function for specificity
            float  recall_pair = calculate_recall(output, mask);
            
            // Update overall metrics (consider averaging across all image-mask pairs)
            this->jaccard_index += jaccard_index_pair;
            this->dice_coefficient += dice_coefficient_pair;
            this->specificity += specificity_pair;
            this->recall += recall_pair;
            
            // Print the report of metrics
            std::cout << "| Image name: " << name << std::endl;
            std::cout << "| Image index: " << img_count << std::endl;
            std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
            std::cout << "| Jaccard Index: " << jaccard_index_pair << std::endl;
            std::cout << "| Dice Coefficient: " << dice_coefficient_pair << std::endl;
            std::cout << "| Specificity: " << specificity_pair << std::endl;
            std::cout << "| Recall: " << recall_pair << std::endl;
            std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
            std::cout << "| Avarage metrics       |" << std::endl;
            std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
            std::cout << "| Jaccard Index: " << this->jaccard_index / (img_count + 1) << std::endl;
            std::cout << "| Dice Coefficient: " << this->dice_coefficient / (img_count + 1) << std::endl;
            std::cout << "| Specificity: " << this->specificity / (img_count + 1) << std::endl;
            std::cout << "| Recall: " << this->recall / (img_count + 1) << std::endl;
            std::cout << "|------------------------------------------------------|" << std::endl;
            std::cout << "| Missing: "<< this->images.size() - (img_count + 1) << std::endl;
            std::cout << "|------------------------------------------------------|" << std::endl;
            
            Result result = Result((int) i, name, output, dice_coefficient_pair, jaccard_index_pair, specificity_pair, recall_pair);
            this->results.push_back(result);
            img_count++;
        }
        
        // Calculate average metrics if iterating through multiple image-mask pairs
        if (this->images.size() > 1) {
            this->jaccard_index /= this->images.size();
            this->dice_coefficient /= this->images.size();
            this->specificity /= this->images.size();
            this->recall /= this->images.size();
        }
        
        // Print or store the calculated metrics (e.g., Jaccard Index, Dice Coefficient, Specificity)
        std::cout << "|----------------------------------------|" << std::endl;
        std::cout << "|-----------------------------------|" << std::endl;
        std::cout << "|------------------------------|" << std::endl;
        std::cout << "| Final results: " << std::endl;
        std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
        std::cout << "| Jaccard Index: " << this->jaccard_index << std::endl;
        std::cout << "| Dice Coefficient: " << this->dice_coefficient << std::endl;
        std::cout << "| Specificity: " << this->specificity << std::endl;
        std::cout << "| Recall: " << this->recall << std::endl;
        std::cout << "|------------------------------|" << std::endl;
        std::cout << "|-----------------------------------|" << std::endl;
        std::cout << "|-------------------------Thanks ;)-----------|" << std::endl;
        std::cout << "|------------------------------------------------------|" << std::endl;
    }
    
    void show_individual_result(int index){
        Result result;
        for(int i = 0; i< this->results.size(); i++){
            if(this->results[i].id == index){
                result = this->results[i];
                break;
            }
        }
        
        Mat output = result.get();
        Mat img = this->images[index];
        Mat mask = this->masks[index];
        string name = this->names[index];
        
        cv::imshow(name, img);
        cv::imshow(name + " mask", mask);
        cv::imshow(name + " result", output);
        
        int key = cv::waitKey(0);

          if (key != -1) {
            cv::destroyAllWindows();
          }
    }
    
    void show_results(){
        std::cout << std::endl;
        std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
        std::cout << "| Final results:        |" << std::endl;
        std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
        std::cout << "| Jaccard Index: " << this->jaccard_index << std::endl;
        std::cout << "| Dice Coefficient: " << this->dice_coefficient << std::endl;
        std::cout << "| Specificity: " << this->specificity << std::endl;
        std::cout << "| Recall: " << this->recall << std::endl;
        std::cout << "|-- -- -- -- -- -- -- --/" << std::endl;
        std::cout << std::endl;
    }
};

#endif /* evaluation_hpp */

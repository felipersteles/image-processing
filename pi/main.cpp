//
//  main.cpp
//  pi
//
//  Created by Felipe Teles on 15/04/24.
//

#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>

#include "lib.h"
#include "utils.hpp"
#include "evaluation.hpp"
#include "pancreas_segmentation.hpp"

using namespace cv;
using namespace std;

void start_menu(Model model){
    int menu = -1, selected_id = -1;
    
    cout 
    << "Com o modelo treinado agora é possível a visualização dos resultados. " << endl
    << "[1] Continuar." << endl
    << "[0] Sair." << endl
    << "Digite uma opção: ";
    
    cin >> menu;
    
    while(menu!=0){
        cout
        << "[1] Visualizar desempenho médio." << endl
        << "[2] Visualizar desemepenho individual." << endl
        << "[0] Sair." << endl
        << "Digite uma opção: ";
        
        cin >> menu;
        
        switch (menu) {
            case 0:
                break;
                
            case 1:
                model.show_results();
                break;
                
            case 2:
                cout<< endl <<"Digite o id do resultado: ";
                cin >> selected_id;
                
                model.show_individual_result(selected_id);
                break;
                
            default:
                cout<< "Digite uma opção válida!" << endl;
                break;
        }
    }
}

Mat perform_segmentation(Mat image){
    
    std::string train_image_folder = "/Users/felipeteles/Development/ufma/dataset/train/img/";
    std::string train_mask_folder = "/Users/felipeteles/Development/ufma/dataset/train/mask/";
    std::vector<cv::Mat> train_images = loadImages(train_image_folder, "Train");
    std::vector<cv::Mat> train_masks = loadImages(train_mask_folder, "Train masks");
    
    std::vector<std::pair<Mat, Mat>> mask_atlas = load_mask_atlas(train_images, train_masks);
    Mat atlas = get_atlas(train_masks);
    
    imshow("Atlas", atlas);
    waitKey(0);
    
    cv::Mat output = map_segmentation(image, mask_atlas, atlas);
    
    return output;
}

int main(int argc, char** argv) {
    
    // perfome the segmentation then evaluate
    Model model = Model();
    start_menu(model);


    // Test image
//    string image_name = "pancreas_004_59.png";
//    cout << endl << "Selecione a imagem: ";
//    cin >> image_name;
//    see_image(image_name);
//
//    cv::Mat pre_img = pre_process_img(img);
//    cv::Mat pos_img = pos_process_img(img);
//    cv::Mat use_mask = reduce_mask(mask);
//    cv::Mat processed_mask = process_mask(pre_img, mask);

//    cv::Mat output;
//    bool find;
//    for (int y = 0; y < pos_img.rows; ++y) {
//        for (int x = 0; x < pos_img.cols; ++x) {
//            if(use_mask.at<uchar>(y,x) > 0 && pos_img.at<uchar>(y,x) > 0 && verify_neighbors(pos_img, y, x)){
//                std::cout << "| [Info]: Pancreas found on pixel(" << x <<", "<< y <<")." << std::endl;
//                output = select_area(pos_img, Point(x, y));
//                find = true;
//                break;
//            }
//        }
//        
//        if(find) break;
//    }
    
//    Mat output = perform_segmentation(img);
//    
//    imshow("Image", img);
//    imshow("Mask", mask);
//    imshow("Pre processamento", pre_img);
//    imshow("Mask processada", processed_mask);
//    imshow("Mask reduzida", use_mask);
//    imshow("Pos processamento", pos_img);
//    imshow("Segmentation", output);
//    
//    waitKey(0);
    
    return 0;
}

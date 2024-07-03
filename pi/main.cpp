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

int main(int argc, char** argv) {
    
    // perfome the segmentation then evaluate
    Model model = Model();
    start_menu(model);

    // Test image
    string image_name;
    cout << endl <<"Selecione a imagem: ";
    cin >> image_name;
    cv::Mat img = cv::imread("/Users/felipeteles/Development/ufma/dataset/test/img/" + image_name, IMREAD_GRAYSCALE);
    cv::Mat mask = cv::imread("/Users/felipeteles/Development/ufma/dataset/test/mask/" + image_name, IMREAD_GRAYSCALE);
    
    cv::Mat pre_img = pre_process_img(img);
    cv::Mat pos_img = pos_process_img(img);
    cv::Mat use_mask = reduce_mask(mask);
    cv::Mat processed_mask = process_mask(pre_img, mask);

    cv::Mat output;
    bool find;
    for (int y = 0; y < pos_img.rows; ++y) {
        for (int x = 0; x < pos_img.cols; ++x) {
            if(use_mask.at<uchar>(y,x) > 0 && use_mask.at<uchar>(y,x) > 0 && verify_neighbors(pos_img, y, x)){
                std::cout << "| [Info]: Pancreas found on pixel(" << x <<", "<< y <<")." << std::endl;
                output = select_area(pos_img, Point(x, y));
                find = true;
                break;
            }
        }
        if(find) break;
    }
    
    imshow("Image", img);
    imshow("Pre processamento", pre_img);
    imshow("Mask", mask);
    imshow("Mask processada", processed_mask);
    imshow("Mask reduzida", use_mask);
    imshow("Pos processamento", pos_img);
    imshow("Segmentation", output);
    
    waitKey(0);
    
    return 0;
}

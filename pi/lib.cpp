//
//  lib.cpp
//  pi
//
//  Created by Felipe Teles on 15/04/24.
//
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <iostream>
#include <vector>
#include <chrono>

#include "lib.h"
#include "connected_components.h"

using namespace cv;
using namespace std;
using namespace cv::ximgproc;
using namespace cv::ximgproc::segmentation;


// Função para quantizar os valores de pixel
Mat quantize_image(Mat img,int maxLevel, int level) {
  Mat quantized = img.clone();

    int interval = maxLevel / level;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
        // Encontrar o nível de quantização para o valor Y
        uchar& pixel = quantized.at<uchar>(i, j, 0);
        pixel = (pixel / interval) * interval + interval / 2;
    }
  }

  return quantized;
}

// Calcula o histograma da imagem
vector<int> calculate_histogram(const Mat &image, int max_lvl) {
    vector<int> histogram(max_lvl, 0);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            histogram[(int)image.at<uchar>(i, j)]++;
        }
    }
    
    return histogram;
}

void draw_histogram(const vector<int>& histogram, string title, int max_level, int height) {
  // Validate input parameters
  if (max_level <= 0 || height <= 0) {
    cerr << "Error: max_level and height must be positive values." << endl;
    return;
  }

  // Calculate histogram size
  int hist_size = max_level;
    
    // Find the maximum value in the histogram (optional, but useful for scaling)
    int max_hist_value = *max_element(histogram.begin(), histogram.end());

    // Avoid division by zero
     float scale_factor = max_hist_value == 0 ? 1.0f : 1.0f / max_hist_value;

    // Create histogram image
    Mat hist_image(height, hist_size, CV_8UC3, Scalar(0, 0, 0));
    
  // Draw histogram bars
  for (int i = 0; i < hist_size; ++i) {
    float bar_height = (histogram[i] * scale_factor * max_hist_value);
      
      Point a = Point(i, height);
      Point b = Point(i, height - cvRound(bar_height));
      
      line(hist_image, a, b, Scalar(200, 200, 200));
  }

  // Display histogram image
  imshow(title, hist_image);
}


// calcula a aparicao dos valores dos pixels
vector<int> caculate_cumulative_histogram(const Mat &image, int max_lvl) {
    vector<int> histogram = calculate_histogram(image, max_lvl);

    vector<int> cumulative_histogram(256, 0);
    cumulative_histogram[0] = histogram[0];
    for (int i = 1; i < max_lvl; i++) {
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i];
    }
    
    return cumulative_histogram;
}

Mat equalize_hist(Mat img){
    
    // criar imagem resultante
    Mat equalized;

    // Aplica a equalização de histograma
    equalizeHist(img, equalized);
    
    return equalized;
}

Mat clahe(Mat img){
    // criar imagem resultante
    Mat equalized;
    
    // Define os parâmetros da equalização de histograma adaptativa
    int tileSize = 8;
    int clipLimit = 32;
    
    cv::Size size(tileSize, clipLimit);
    
    // Aplica a equalização de histograma adaptativa
    cv::createCLAHE(clipLimit, size)->apply(img, equalized);
    
    return equalized;
}

// Aplica a especificação do histograma com base num histograma de referencia
Mat match_histograms(const Mat &target, const vector<int> &source_cumalative_histogram, int max_lvl) {

    // calcula o histograma da imagem alvo
    vector<int> target_hist = calculate_histogram(target, max_lvl);
    
    vector<uchar> lut(max_lvl, 0);

    // 1: comparação do valor das probabilidades
    for (int i = 0; i < max_lvl; i++) {
        int j = 0;
        double min_diff = abs(target_hist[i] - source_cumalative_histogram[j]);
        for (int k = 0; k < max_lvl; k++) {
            double diff = abs(target_hist[i] - source_cumalative_histogram[k]);
            if (diff < min_diff) {
                min_diff = diff;
                j = k;
            }
        }
        
        lut[i] = uchar(j);
    }
    
    // criar imagem resultante
    Mat matched = target.clone();
    
    // 2: aplicando a especificação
    for (int i = 0; i < target.rows; i++) {
        for (int j = 0; j < target.cols; j++) {
            matched.at<uchar>(i, j) = lut[matched.at<uchar>(i, j)];
        }
    }
    
    return matched;
}

// função que gera o kernel gaussiano com base em um desvio padrão
Mat gauss_kernel(int kernel_size_x, int kernel_size_y, float sigma){
    
    float sum_term {0};
    Mat kernel = Mat::zeros(kernel_size_x, kernel_size_y, CV_32F);
    
    // criar kernel
    // formula do kernel: G(x, y) = (1 / (2 * M_PI * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2)) -> (fonte: gemini do google)
    for (int i = -kernel_size_y/2; i <= kernel_size_y/2; i++)
    {
        for (int j = -kernel_size_x/2; j <= kernel_size_x/2; j++)
        {
            kernel.at<float>(i + kernel_size_y/2, j + kernel_size_x/2) = exp(-(pow(i,2) + pow(j,2))/(2*pow(sigma,2)));
            sum_term += kernel.at<float>(i + kernel_size_y/2,j + kernel_size_x/2);
        }
    }
    
    kernel /= sum_term;
    
    return kernel;
}

// função que aplica o kernel em uma imagem
Mat apply_kernel(Mat img, Mat kernel){
    Mat result;
    
    // função nativa do opencv
    filter2D(img, result, -1, kernel);
    
    return result;
}

// função que aplica o otsu threshhold em uma imagem
Mat otsu_threshold(Mat img){
    Mat result;
    
    // função nativa do opencv
    threshold(img, result, 0, 255, THRESH_BINARY | THRESH_OTSU);
    
    return result;
}

Mat apply_blur(Mat image){
    
    Mat blur = image.clone();
    
    GaussianBlur(image, blur, Size(3,3), 0);
    
    return blur;
}

Mat get_sobel(Mat img_blur, SobelDirection dir){
    Mat sobel;
    
    switch (dir) {
        case x:
            Sobel(img_blur, sobel, CV_64F, 1, 0, 5);
            break;
            
        case y:
            Sobel(img_blur, sobel, CV_64F, 0, 1, 5);            
            break;
            
        default:
            Sobel(img_blur, sobel, CV_64F, 1, 1, 5);
            break;
    }
    
    return sobel;
}

Mat get_canny(Mat img_blur){
    // Canny edge detection
    Mat edges;
    
    Canny(img_blur, edges, 100, 200, 3, false);
    
    return edges;
}

Mat get_laplacian(Mat img){
    
    // Canny laplacian detection
    Mat laplacian;
    Laplacian(img, laplacian, CV_64F);
    
    return laplacian;
}

Mat get_gradient(Mat img){
    // Create a structuring element
        int morph_size = 2;
        Mat element = getStructuringElement(
            MORPH_RECT,
            Size(2 * morph_size + 1,
                 2 * morph_size + 1),
            Point(morph_size,
                  morph_size));
    
        Mat output;
      
        // Gradient
        morphologyEx(img, output,
                     MORPH_GRADIENT, element,
                     Point(-1, -1), 1);
    
    return output;
}


int yPrewittGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y - 1, x - 1) +
         image.at<uchar>(y - 1, x) +
        image.at<uchar>(y - 1, x + 1) -
        image.at<uchar>(y + 1, x - 1) -
        image.at<uchar>(y + 1, x) -
        image.at<uchar>(y + 1, x + 1);
}

int xPrewittGradient(cv::Mat image, int x, int y)
{
    return image.at<uchar>(y - 1, x - 1) +
         image.at<uchar>(y, x - 1) +
        image.at<uchar>(y + 1, x - 1) -
        image.at<uchar>(y - 1, x + 1) -
        image.at<uchar>(y, x + 1) -
        image.at<uchar>(y + 1, x + 1);
}

Mat get_prewitt(Mat img){
    Mat dstPrewitt = img.clone();
    
    for (int y = 0; y < img.rows; y++)
            for (int x = 0; x < img.cols; x++)
                dstPrewitt.at<uchar>(y, x) = 0.0;
            
    
    int gpx,gpy,sump;
    
    for (int y = 1; y < img.rows - 1; y++) {
        for (int x = 1; x < img.cols - 1; x++) {
            
            //Prewitt
            gpx = xPrewittGradient(img, x, y);
            gpy = yPrewittGradient(img, x, y);
            sump = sqrt(powf(gpx, 2.0) + powf(gpy, 2.0));
            sump = sump > 255 ? 255 : sump;
            sump = sump < 0 ? 0 : sump;
            dstPrewitt.at<uchar>(y, x) = sump;
        }
    }
    
    return dstPrewitt;
}

Mat apply_kmeans(Mat img){
    Mat src = img.clone();
    
    Mat afterOtsu = otsu_threshold(src);
    
    Mat samples(afterOtsu.rows * afterOtsu.cols, 3, CV_32F);
    for (int y = 0; y < afterOtsu.rows; y++)
        for (int x = 0; x < afterOtsu.cols; x++)
            for (int z = 0; z < 3; z++)
                samples.at<float>(y + x*afterOtsu.rows, z) = afterOtsu.at<Vec3b>(y, x)[z];


    int clusterCount = 5;
    Mat labels;
    int attempts = 5;
    Mat centers;
    kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::EPS + cv::TermCriteria::COUNT, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);


    Mat res(afterOtsu.size(), afterOtsu.type());
    for (int y = 0; y < afterOtsu.rows; y++)
        for (int x = 0; x < afterOtsu.cols; x++)
        {
            int cluster_idx = labels.at<int>(y + x*afterOtsu.rows, 0);
            res.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
            res.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
            res.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
        }
    
    return res;
}

void relabelSuperpixels(cv::Mat &labels) {

    int max_label = 0;
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            if (labels.at<int>(i, j) > max_label) {
                max_label = labels.at<int>(i, j);
            }
        }
    }

    int current_label = 0;
    vector<int> label_correspondence(max_label + 1, -1);

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels.at<int>(i, j);

            if (label_correspondence[label] < 0) {
                label_correspondence[label] = current_label++;
            }

            labels.at<int>(i, j) = label_correspondence[label];
        }
    }
}


Mat apply_superpixel(Mat img){
    
    Mat result = img.clone();
    
    // Set SLIC parameters
    int region_size = 100; // Adjust for desired superpixel size
    float regularity = 50.0; // Adjust for shape compactness
    int num_iterations = 5;

    // Create SLIC object
    Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(result,SLIC,region_size,regularity);
    slic->iterate(num_iterations);
    
    Mat labels;
    slic->getLabels(labels);
    
    double t = (double) getTickCount();
    
    t = ((double) getTickCount() - t) / getTickFrequency();
    cout << "SLIC"
         << " segmentation took " << (int) (t * 1000)
         << " ms with " << slic->getNumberOfSuperpixels() << " superpixels" << endl;
    
    Mat mask;
    // get the contours for displaying
    slic->getLabelContourMask(mask, true);
    result.setTo(Scalar(0, 0, 255), mask);
    
    return result;
}

vector<KeyPoint> get_features_using_SIFT(Mat img){
    vector<KeyPoint> keypoints;
    Ptr<Feature2D> f2d_detector = SIFT::create();
    f2d_detector->detect(img, keypoints);
    
    // Add results to image and save.
    Mat output;
    drawKeypoints(img, keypoints, output);
    imshow("sift result", output);
    
    return keypoints;
}

void pancreas_segmentation(Mat img, Mat mask){
    cout << "equalizar histograma" << endl
    <<  "aplicar extracao de caracteristicas baseada em caminho" << endl
    << "aplicar o region growing ou tecnica de segmentacao" << endl
    << "aplicar metricas"<< endl;
    
    // Exibe os resultados
    imshow("Imagem Original", img);
    // Calcula o histograma
    vector<int> histograma = calculate_histogram(img);
    draw_histogram(histograma, "Histograma da imagem original");
    
    // Create a Mat to store the XOR result
    Mat xor_result; // 8UC1 for single-channel (grayscale)

      // Perform element-wise XOR operation using bitwise_xor
    bitwise_xor(img, mask, xor_result);

    imshow("Imagem depois do XOR", xor_result);
    
    Mat equalizedImg = clahe(img);
    imshow("Equalizada", equalizedImg);
    
    vector<int> newHistograma = calculate_histogram(equalizedImg);
    draw_histogram(newHistograma, "Histograma da imagem equalizada");
    
    // Aplicar superpixel na imagem equalizada
    Mat blurImg = apply_blur(equalizedImg);
    Mat superPixelImg = apply_superpixel(img);
    imshow("Imagem depois do superpixel", superPixelImg);
    
    get_features_using_SIFT(superPixelImg);
}

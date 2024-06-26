//
//  lib.cpp
//  pi
//
//  Created by Felipe Teles on 15/04/24.
//
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/types.hpp>

#include <iostream>
#include <vector>
#include <chrono>

#include "lib.h"

using namespace cv;
using namespace std;
using namespace cv::ximgproc;
using namespace cv::ximgproc::segmentation;
using namespace cv::xfeatures2d;


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

Mat clahe(Mat img, int tileSize, int clipLimit){
    // criar imagem resultante
    Mat equalized;
    
    // Define os parâmetros da equalização de histograma adaptativa
    
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

Mat unsharp(Mat image, float amount) {
    
    Mat blurred_image = apply_blur(image);
    // Calculate the mask (difference between original and blurred)
    Mat mask = blurred_image - image;
    
    // Enhance the mask for better sharpening effect
    mask = mask * amount;  // Adjust the amplification factor
    
    // Sharpen the image by adding the weighted mask
    Mat sharpened_image = image + mask;
    
    // Clip pixel values to avoid overflow/underflow
    convertScaleAbs(sharpened_image, sharpened_image, 1.0, 0.0);
    
    return sharpened_image;
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

Mat get_canny(Mat img, double threshold1, double threshold2){
    Mat img_blur = apply_blur(img);
    
    // Canny edge detection
    Mat edges;
    
    Canny(img_blur, edges, threshold1, threshold2, 3, false);
    
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


Mat apply_superpixel(Mat img,  int region_size, float regularity, int num_iterations){
    
    Mat result = img.clone();
    
    // Create SLIC object
    Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(result,SLIC,region_size,regularity);
    slic->iterate(num_iterations);
    
    Mat labels;
    slic->getLabels(labels);
    
    double t = (double) getTickCount();
    
//    t = ((double) getTickCount() - t) / getTickFrequency();
//    cout << "SLIC"
//    << " segmentation took " << (int) (t * 1000)
//    << " ms with " << slic->getNumberOfSuperpixels() << " superpixels" << endl;
    
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

Mat extract_SIFT_features(const cv::Mat& image) {
    // Create SIFT feature detector and extractor
    Ptr<Feature2D> sift = SIFT::create();
    
    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    
    return descriptors;
}

Mat extract_SURF_features(const cv::Mat& image) {
    // Create SIFT feature detector and extractor
    Ptr<Feature2D> sift = SURF::create();
    
    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    
    return descriptors;
}

cv::Mat vessel_enhancement(const cv::Mat& image, double sigma1, double sigma2) {
    // Convert to grayscale if necessary
    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image.clone();
    }
    
    // Apply Gaussian filtering with two scales for vessel enhancement
    cv::Mat filtered1, filtered2;
    cv::GaussianBlur(gray_image, filtered1, cv::Size(0, 0), sigma1); // Larger sigma for background suppression
    cv::GaussianBlur(gray_image, filtered2, cv::Size(0, 0), sigma2); // Smaller sigma for preserving vessel details
    
    // Vessel enhancement using difference of Gaussians (DoG)
    cv::Mat vessel_enhanced = filtered1 - filtered2;
    
    // Apply normalization (optional, adjust based on your needs)
    cv::Mat normalized;
    cv::normalize(vessel_enhanced, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    return normalized;
}

cv::Mat multi_scale_line_enhancement(const cv::Mat& image, double sigmaMin, double sigmaMax) {
    // Input image validation
    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return image;
    }
    
    // Convert to grayscale if needed (assuming BGR format)
    cv::Mat grayImage = image.clone();
    
    // Prepare filtered image
    cv::Mat filteredImage = cv::Mat::zeros(grayImage.size(), CV_32F);
    
    // Loop through different scales (sigma values)
    for (double sigma = sigmaMin; sigma <= sigmaMax; sigma += 1.0) {
        // Create Gaussian filter kernel
        cv::Mat kernel = cv::getGaussianKernel(5, sigma, CV_32F);
        
        // Apply derivative of Gaussian filter in x and y directions
        cv::Mat filteredX, filteredY;
        cv::filter2D(grayImage, filteredX, CV_32F, kernel, cv::Point(-1, 0));
        cv::filter2D(grayImage, filteredY, CV_32F, kernel.t(), cv::Point(0, -1));
        
        // Combine responses (consider other options for combining)
        cv::Mat response = filteredX.mul(filteredX) + filteredY.mul(filteredY);
        
        // Accumulate filtered response across scales
        filteredImage += response;
    }
    
    // Normalize and convert to final image format (consider scaling factor)
    cv::normalize(filteredImage, filteredImage, 0.0, 1.0, cv::NORM_MINMAX);
    filteredImage.convertTo(filteredImage, image.type());
    
    return filteredImage;
}


cv::Mat apply_transformation(const cv::Mat& image, const cv::Mat& transformation) {
    // Check transformation size
    if (transformation.cols != 3 || transformation.rows != 3) {
        std::cerr << "Error: Transformation matrix must be 3x3." << std::endl;
        return image;
    }
    
    cv::Mat transformed_image;
    cv::warpAffine(image, transformed_image, transformation, image.size());
    return transformed_image;
}

Mat trasnform_image(const cv::Mat& image){
    Mat transformation = image.clone();
    
    // Define transformation matrices
    cv::Mat rigid_transform = cv::Mat::zeros(3, 3, CV_64FC1);
    cv::Mat affine_transform = cv::Mat::zeros(3, 3, CV_64FC1);
    cv::Mat nonlinear_transform; // Define nonlinear transformation logic (example not provided)
    
    // Rigid transformation: Rotation (45 degrees) and translation (20 pixels right, 10 pixels down)
    double angle_rad = CV_PI / 4; // 45 degrees in radians
    rigid_transform = (cv::Mat_<double>(3, 3) <<
                       cos(angle_rad), -sin(angle_rad), 20,
                       sin(angle_rad),  cos(angle_rad), 10,
                       0,              0,              1);
    
    // Affine transformation: Scaling (1.2x in x, 0.8x in y), shearing (10 degrees), and translation (30 pixels right, 5 pixels up)
    affine_transform = (cv::Mat_<double>(3, 3) <<
                        1.2,  0.2 * tan(CV_PI / 18 * 10), 30,
                        0,    0.8,                         5,
                        0,              0,              1);
    
    // Apply transformations and display results
    cv::Mat rigid_image = apply_transformation(image.clone(), rigid_transform);
    //    cv::Mat affine_image = applyTransformation(image.clone(), affine_transform);
    
    return rigid_image;
}

// Function to perform erosion
Mat erode(const Mat& image, Mat& kernel, int iterations) {
    // Input image validation
    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return image;
    }
    
    // Anchor point default (center of the kernel)
    Point anchor = Point(-1, -1);
    
    // Erosion operation
    Mat eroded_image;
    erode(image, eroded_image, kernel, anchor, iterations, BORDER_REPLICATE);
    
    return eroded_image;
}

// Function to perform dilation
Mat dilate(const Mat& image, Mat& kernel, int iterations) {
    // Input image validation
    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return image;
    }
    
    // Anchor point default (center of the kernel)
    Point anchor = Point(-1, -1);
    
    // Dilation operation
    Mat dilated_image;
    dilate(image, dilated_image, kernel, anchor, iterations, BORDER_REPLICATE);
    
    return dilated_image;
}

Mat closing(const Mat& image, Mat& kernel, int iterations) {
    // Input image validation
    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return image;
    }
    
    // Anchor point default (center of the kernel)
    Point anchor = Point(-1, -1);
    
    // Dilation followed by erosion
    Mat dilated_image;
    dilate(image, dilated_image, kernel, anchor, iterations, BORDER_REPLICATE);
    
    Mat closed_image;
    erode(dilated_image, closed_image, kernel, anchor, iterations, BORDER_REPLICATE);
    
    return closed_image;
}

cv::Mat opening(const cv::Mat& image, const cv::Mat& kernel, int iterations) {
    // Input image validation
    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return image;
    }
    
    // Anchor point default (center of the kernel)
    cv::Point anchor = cv::Point(-1, -1);
    
    // Erosion followed by dilation
    cv::Mat eroded_image;
    cv::erode(image, eroded_image, kernel, anchor, iterations, cv::BORDER_REPLICATE);
    
    cv::Mat opened_image;
    cv::dilate(eroded_image, opened_image, kernel, anchor, iterations, cv::BORDER_REPLICATE);
    
    return opened_image;
}


double calculate_log_likelihood(const cv::Mat& background_cov, const cv::Mat& background_cov_mean,const cv::Scalar& pixel_value, const cv::Mat& mask) {
    // Error handling: check input dimensions and types
    if (background_cov.rows != background_cov.cols) {
        return -INFINITY; // Return negative infinity for invalid input
    }
    
    // Calculate determinant of the covariance matrix
    double determinant = cv::determinant(background_cov);
    if (determinant <= 0) {
        return -INFINITY; // Return negative infinity for non-positive definite covariance
    }
    
    // Extract mean of background pixels from the mean vector (assuming column storage)
    cv::Mat mean_background = cv::Mat::zeros(2, 1, background_cov.type());
    mean_background.at<double>(0, 0) = background_cov_mean.at<double>(0, 0); // Assuming mean is stored in the first row and column
    
    // Assuming pixel_value is a background pixel within the mask
    cv::Mat centered_pixel = pixel_value - mean_background;
    
    // Calculate Mahalanobis distance using matrix multiplication for efficiency
    cv::Mat mahalanobis_dist_mat = centered_pixel.t() * background_cov.inv() * centered_pixel;
    double mahalanobis_dist = mahalanobis_dist_mat.at<double>(0, 0); // Extract single value
    
    // Update log-likelihood (assuming independent samples)
    return -0.5 * (mahalanobis_dist + std::log(2.0 * M_PI * determinant));
}


cv::Mat calculate_bayer_probs(const cv::Mat& image, const cv::Mat& mask,
                              double foreground_prob_threshold,
                              double background_prob_threshold,
                              double default_foreground_prob) {
    
    // Error handling for input matrix dimensions and types
    if (image.size() != mask.size()) {
        std::cerr << "| [Error]: Invalid input matrix dimensions or types!" << std::endl;
        return cv::Mat(); // Return empty Mat on error
    }
    
    
    cv::Mat image_data = image.reshape(1, (int) image.total());
    cv::Mat mask_data = mask.reshape(1, (int) mask.total());
    
    // Stack data together (assuming correspondence between pixels)
    Mat data;
    cv::hconcat(image_data, mask_data, data);
    
    cv::Mat background_cov, background_cov_mean;
    cv::calcCovarMatrix(data, background_cov, background_cov_mean, COVAR_ROWS | COVAR_NORMAL);
    
    cv::Mat bayer_probs = cv::Mat::zeros(image.size(), CV_64FC1);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (mask.at<uchar>(y, x) > 0) { // Pixel belongs to foreground (object)
                bayer_probs.at<double>(y, x) = default_foreground_prob; // Assign probability for foreground
            } else {
                // Calculate probability using multivariate Gaussian with the provided covariance matrix
                Scalar pixel_value = image.at<cv::Vec3d>(y, x);
                double log_likelihood = calculate_log_likelihood(background_cov, background_cov_mean,pixel_value, mask);
                
                // Convert log-likelihood to probability (optional, adjust based on your needs)
                double probability = std::exp(log_likelihood); // You might want to use a threshold or normalization
                bayer_probs.at<double>(y, x) = std::max(probability, background_prob_threshold);
            }
        }
    }
    
    return bayer_probs;
}

// Use recursive function to explore connected components
void exploreConnected(int x, int y, Mat& segmented_image) {
    if (x < 0 || x >= segmented_image.cols ||
        y < 0 || y >= segmented_image.rows ||
        segmented_image.at<uchar>(y, x) == 0) { // Background or already processed
        return;
    }
    
    segmented_image.at<uchar>(y, x) = 0; // Mark as processed (background)
    
    // Explore neighbors recursively (4-neighborhood)
    exploreConnected(x - 1, y,segmented_image);
    exploreConnected(x + 1, y,segmented_image);
    exploreConnected(x, y - 1,segmented_image);
    exploreConnected(x, y + 1,segmented_image);
}

Mat select_area(const Mat& img, const Point& seed_point) {
    
    Mat binary_image = get_canny(img, 100, 200);
    
    // Check if input is a single-channel binary image
    if (binary_image.channels() != 1 || binary_image.depth() != CV_8U) {
        std::cerr << "Error: Input image must be a single-channel binary image (CV_8UC1)" << std::endl;
        return Mat();
    }
    
    // Check if seed point is within image bounds
    if (seed_point.x < 0 || seed_point.x >= binary_image.cols ||
        seed_point.y < 0 || seed_point.y >= binary_image.rows) {
        std::cerr << "Error: Seed point is outside image bounds!" << std::endl;
        return Mat();
    }
    
    Mat segmented_image = binary_image.clone();
    
//    imshow("binary_image", binary_image);
//    waitKey(0);
    
    // Flood fill starting from the seed point
    floodFill(segmented_image, seed_point, 255);
    
//    imshow("floodFill", segmented_image);
//    waitKey(0);
    
    // Invert the segmentation result to remove the filled area (becomes 0)
    bitwise_not(segmented_image, segmented_image);
    
    // Combine the inverted segmentation with the original image
    // to get the desired result (only seed pixel's area remains)
    bitwise_xor(binary_image, segmented_image,segmented_image);
    bitwise_not(segmented_image, segmented_image);
    
    // fix mask
    Mat closing_kernel = Mat::ones(5,5, CV_8U);
    Mat result = closing(segmented_image, closing_kernel, 2);
    
    return result;
}

// Function to check if a pixel coordinate is within image boundaries
bool inBounds(const Mat& image, int row, int col) {
    return (row >= 0 && row < image.rows) && (col >= 0 && col < image.cols);
}

bool verify_neighbors(const Mat& image, int row, int col) {
    bool allNeighbors255 = false;
    
    // Define offsets for all neighbors
    int offsets[][2] = {
        {-1, 0},  // Top neighbor
        {1, 0},   // Bottom neighbor
        {0, -1},  // Left neighbor
        {0, 1},   // Right neighbor
        {-1, -1}, // Top-left neighbor
        {-1, 1},  // Top-right neighbor
        {1, -1},  // Bottom-left neighbor
        {1, 1}    // Bottom-right neighbor
    };
    
    // Check if all neighbors need to be 255 (default) or any can be non-255
    bool anyNon255 = false;
    
    for (int i = 0; i < 8; ++i) {
        int neighbor_row = row + offsets[i][0];
        int neighbor_col = col + offsets[i][1];
        
        // Check if neighbor is within image bounds
        if (inBounds(image, neighbor_row, neighbor_col)) {
            // Access pixel value (adjust for data type if needed)
            uchar pixel_value = image.at<uchar>(neighbor_row, neighbor_col);
            if (pixel_value != 255) {
                anyNon255 = true;
                break; // Early exit if any neighbor is not 255 (depending on your requirement)
            }
        } else {
            // Consider out-of-bounds pixels as non-255 for strict checking (modify if needed)
            anyNon255 = true;
            break; // Early exit if any neighbor is out of bounds (depending on your requirement)
        }
    }
    
    return !anyNon255;
}

// Function to create a circular structuring element (kernel)
Mat create_circular_kernel(int radius) {
    // Ensure radius is positive and odd
    if (radius <= 0 || radius % 2 == 0) {
        CV_Error(cv::Error::StsBadArg, "Radius must be a positive odd integer.");
    }
    
    // Calculate diameter
    int diameter = radius * 2 + 1;
    
    // Create a square Mat filled with zeros
    Mat kernel = Mat::zeros(diameter, diameter, CV_8U);
    
    // Calculate center coordinates
    int center_x = diameter / 2;
    int center_y = center_x;
    
    // Set elements within the circle to 1 (modify for different shapes if needed)
    for (int i = 0; i < diameter; ++i) {
        for (int j = 0; j < diameter; ++j) {
            double distance = sqrt(pow(i - center_x, 2) + pow(j - center_y, 2));
            if (distance <= radius) {
                kernel.at<uchar>(i, j) = 1;
            }
        }
    }
    
    return kernel;
}


#ifndef SIFT_H
#define SIFT_H
#include <opencv2/core.hpp>

typedef float sift_wt;

class SIFT{

  private:
    int firstOctave;
    
    double sigma;

    int nOctaveLayers;

    int nOctaves;

    double contrastThreshold;

    double edgeThreshold;

  public:

    SIFT();

    SIFT(int firstOctave, double sigma, int nOctaveLayers);

    ~SIFT(){};

    void createInitialEdge(cv::Mat_<float>& edge, bool doubleImageSize, cv::Mat_<float>& base);

    void buildGaussianPyramid(const cv::Mat_<float>& base, std::vector<cv::Mat>& pyr);

    void debugGaussianPyramid(std::vector<cv::Mat>& gauss_pyrX, std::vector<cv::Mat>&
        gauss_pyrY, std::string debug_prefix);

    void buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr);

    void debugDoGPyramid(std::vector<cv::Mat>& gauss_pyrX, std::vector<cv::Mat>& gauss_pyrY,
        std::vector<cv::Mat>& dog_pyrX, std::vector<cv::Mat>& dog_pyrY, std::string debug_prefix);

    void debugExtrema(const std::vector<cv::Mat>& gauss_pyrX, const std::vector<cv::Mat>& gauss_pyrY,
        const std::vector<cv::Mat>& dog_pyrX, const std::vector<cv::Mat>& dog_pyrY, 
        int o, int i, // octave, gaussian level
        int r, int c); // (r,c) edge point idx

    void findScaleSpaceExtrema(
        const std::vector<cv::Mat>& gauss_pyrX,  
        const std::vector<cv::Mat>& gauss_pyrY, 
        const std::vector<cv::Mat>& dog_pyrX,
        const std::vector<cv::Mat>& dog_pyrY, 
        const std::vector<cv::Mat>& dog_pyr,
        std::vector<cv::KeyPoint>& keypoints);

    void detectAndCompute(cv::Mat_<float>& edge, std::vector<cv::KeyPoint>&
        keypoints, int nfeatures, std::string debug_prefix);
};
#endif

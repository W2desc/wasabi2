#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/features2d.hpp>

#include "feature/sift.h"
#include "util/edge.h"
#include "util/cst.h"

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 2;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

static const int SIFT_FIXPT_SCALE = 1;


SIFT::SIFT(){
  firstOctave = -1;
  sigma = 1.6;
  nOctaveLayers = 3;
  nOctaves = 3;
  contrastThreshold = 0.02;
  edgeThreshold = 10;
}


SIFT::SIFT(int firstOctave, double sigma, int nOctaveLayers) : 
  firstOctave(firstOctave), 
  sigma(sigma), 
  nOctaveLayers(nOctaveLayers){

}


void SIFT::createInitialEdge(cv::Mat_<float>& edge, bool doubleImageSize, cv::Mat_<float>& base){
  float factor = 2;
  float sig_diff = 0;
  if (doubleImageSize){
    resizeEdge(edge, factor, base, CV_INTER_LINEAR);
    sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01) );
  }
  else{
    base = edge.clone();
    sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01) );
  }
  //printf("sig_diff: %f\n", sig_diff);
  cv::GaussianBlur(base, base, cv::Size(), sig_diff, sig_diff);
}


void SIFT::buildGaussianPyramid(const cv::Mat_<float>& base, std::vector<cv::Mat>& pyr){
  //printf("buildGaussianPyramid: %d\nnOctaveLayers: %d\n", nOctaves, nOctaveLayers);
  // you can see pyr as a matrix on [nOctaves, nOctavesLayers]
  std::vector<double> sig(nOctaveLayers + 3); // size = (3+3)
  pyr.resize(nOctaves*(nOctaveLayers + 3)); // size = 7*(3+3) 

  // precompute Gaussian sigmas using the following formula:
  //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
  sig[0] = sigma; // 1.6 ~ sqrt(2)
  //double k = std::pow( 2., 1. / nOctaveLayers );
  for( int i = 1; i < nOctaveLayers + 3; i++ ){ // for i=1<(3+3) i.e. 5 times
    //double sig_prev = std::pow(k, (double)(i-1))*sigma;
    //double sig_total = sig_prev*k;
    //sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    sig[i] = sigma*sig[i-1]; // because it is the reccurence relation that best approximates the laplacian
  }

  for( int o = 0; o < nOctaves; o++ ){ // 0->6
    for( int i = 0; i < nOctaveLayers + 3; i++ ){ // 0->5
      //printf("buildGaussianPyramid:: o: %d\ti: %d\n", o, i);
      cv::Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
      if( o == 0  &&  i == 0 ){
        //printf("buildGaussianPyramid:: base.rows: %d\tbase.cols: %d\n", base.rows, base.cols);
        dst = base;
      }
      // base of new octave is halved image from end of previous octave
      else if( i == 0 ){
        const cv::Mat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
        resizeEdge(src, 0.5, dst, cv::INTER_NEAREST); // subsample
        //printf("buildGaussianPyramid:: src.rows: %d\tsrc.cols: %d\n", src.rows, src.cols);
        //printf("buildGaussianPyramid:: dst.rows: %d\tdst.cols: %d\n", dst.rows, dst.cols);
      }
      else{
        const cv::Mat& src = pyr[o*(nOctaveLayers + 3) + i-1];
        cv::GaussianBlur(src, dst, cv::Size(), sig[i], sig[i]);
      }
    }
  }
}


void SIFT::debugGaussianPyramid(std::vector<cv::Mat>& gauss_pyrX, std::vector<cv::Mat>&
    gauss_pyrY, std::string debug_prefix){
  for( int o = 0; o < nOctaves; o++ ){ // 0->6
    for( int i = 0; i < nOctaveLayers + 3; i++ ){ // 0->5
      cv::Mat& edgeX = gauss_pyrX[o*(nOctaveLayers + 3) + i];
      cv::Mat& edgeY = gauss_pyrY[o*(nOctaveLayers + 3) + i];
      //printf("o: %d\ti: %d\tedgeX.rows: %d\tedgeX.cols: %d\n", o, i, edgeX.rows, edgeX.cols);
      //printf("o: %d\ti: %d\tedgeY.rows: %d\tedgeY.cols: %d\n\n", o, i, edgeY.rows, edgeY.cols);
      
      //// to file
      //char buff[512];
      //sprintf(buff, "%s_gauss_pyr_%d_%d.txt", debug_prefix.c_str(), o, i);
      ////printf("debugGaussianPyramid::buff: %s\n", buff);
      //FILE* fp = fopen(buff, "w");
      //fprintf(fp, "# edgeX, edgeY\n");
      //for (int k=0; k<edgeX.cols; k++){
      //  fprintf(fp, "%f %f\n", edgeX.at<float>(k), edgeY.at<float>(k));
      //}
      //fclose(fp);

      if (o==0){ 
        cv::Mat_<uint8_t> img2(2*IMG_H, 2*IMG_W, (uint8_t) 0);
        edge2img(edgeX, edgeY, img2);
        cv::imshow("img", img2);
        cv::waitKey(0);
      }
      else{
        cv::Mat_<uint8_t> img(IMG_H, IMG_W, (uint8_t) 0);
        edge2img(edgeX, edgeY, img);
        cv::imshow("img", img);
        cv::waitKey(0);
      }
    }
  }
}


class buildDoGPyramidComputer : public cv::ParallelLoopBody {
  public:
    buildDoGPyramidComputer(
        int _nOctaveLayers,
        const std::vector<cv::Mat>& _gpyr,
        std::vector<cv::Mat>& _dogpyr)
      : nOctaveLayers(_nOctaveLayers),
      gpyr(_gpyr),
      dogpyr(_dogpyr) { }

    void operator()( const cv::Range& range ) const override{
      const int begin = range.start;
      const int end = range.end;

      for( int a = begin; a < end; a++){
        const int o = a / (nOctaveLayers + 2);
        const int i = a % (nOctaveLayers + 2);

        const cv::Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
        const cv::Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
        cv::Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
        subtract(src2, src1, dst, cv::noArray(), cv::DataType<float>::type);
      }
    }

  private:
    int nOctaveLayers;
    const std::vector<cv::Mat>& gpyr;
    std::vector<cv::Mat>& dogpyr;
};


void SIFT::buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr){
  dogpyr.resize( nOctaves*(nOctaveLayers + 2) );
  cv::parallel_for_(cv::Range(0, nOctaves * (nOctaveLayers + 2)), buildDoGPyramidComputer(nOctaveLayers, gpyr, dogpyr));
}


void SIFT::debugDoGPyramid(std::vector<cv::Mat>& gauss_pyrX, std::vector<cv::Mat>& gauss_pyrY,
    std::vector<cv::Mat>& dog_pyrX, std::vector<cv::Mat>& dog_pyrY, std::string debug_prefix){

  for(int a=0; a<nOctaves*(nOctaveLayers+2); a++){
    const int o = a / (nOctaveLayers + 2);
    const int i = a % (nOctaveLayers + 2);
      
    const cv::Mat_<float>& dogX = dog_pyrX[o*(nOctaveLayers + 2) + i];
    const cv::Mat_<float>& dogY = dog_pyrY[o*(nOctaveLayers + 2) + i];
    const cv::Mat_<float>& edgeX = gauss_pyrX[o*(nOctaveLayers + 3) + i];
    const cv::Mat_<float>& edgeY = gauss_pyrY[o*(nOctaveLayers + 3) + i];

    char buff[512];
    sprintf(buff, "%s_dog_pyr_%d_%d.txt", debug_prefix.c_str(), o, i);
    //printf("debugDOGPyramid::buff: %s\n", buff);
    FILE* fp = fopen(buff, "w");
    fprintf(fp, "# edgeX, edgeY, dogX, dogY\n");
    for (int k=0; k<dogX.cols; k++){
      fprintf(fp, "%f %f %f %f\n", edgeX(0,k), edgeY(0,k), dogX(0,k), dogY(0,k));
    }
    fclose(fp);

    //int dogImgH=0, dogImgW=0;
    //if (o==0){
    //  dogImgH = 2*IMG_H;
    //  dogImgW = 2*IMG_W;
    //}
    //else{
    //  dogImgH = IMG_H;
    //  dogImgW = IMG_W;
    //}
    //cv::Mat_<float> dogImg(dogImgH, dogImgW, (float) 0);
    //for (int j=0; j<edgeY.cols; j++){
    //  int x = (int) floor(edgeX(0,j));
    //  int y = (int) floor(edgeY(0,j));
    //  float gradX = dogX(0,j);
    //  float gradY = dogY(0,j);
    //  //dogImg(y,x) = gradX*gradX + gradY*gradY;
    //  dogImg(y,x) = gradX + gradY;
    //}

    //// visualise it
    //if(o==1){ // only at original img size
    //  dogImg = cv::abs(dogImg);
    //  double min, max;
    //  cv::minMaxLoc(dogImg, &min, &max);
    //  //printf("min: %f\tmax: %f\n", min, max);
    //  dogImg /= max;
    //  dogImg *= 255;
    //  cv::Mat_<uint8_t> dogImgU;
    //  dogImg.convertTo(dogImgU, CV_8UC1);
    //  cv::imshow("dogImg", dogImgU);
    //  cv::waitKey(0);
    //}
  }
}


void SIFT::debugExtrema(const std::vector<cv::Mat>& gauss_pyrX, const std::vector<cv::Mat>& gauss_pyrY,
    const std::vector<cv::Mat>& dog_pyrX, const std::vector<cv::Mat>& dog_pyrY, 
    int o, int i, // octave, gaussian level
    int r, int c){ // (r,c) edge point idx

  //const int o = a / (nOctaveLayers + 2);
  //const int i = a % (nOctaveLayers + 2);

  cv::Mat_<float> dogX = dog_pyrX[o*(nOctaveLayers + 2) + i];
  cv::Mat_<float> dogY = dog_pyrY[o*(nOctaveLayers + 2) + i];
  const cv::Mat_<float> edgeX = gauss_pyrX[o*(nOctaveLayers + 3) + i];
  const cv::Mat_<float> edgeY = gauss_pyrY[o*(nOctaveLayers + 3) + i];

  int dogImgH=0, dogImgW=0;
  if (o==0){
    dogImgH = 2*IMG_H;
    dogImgW = 2*IMG_W;
  }
  else{
    dogImgH = IMG_H;
    dogImgW = IMG_W;
  }
  cv::Mat_<float> dogImg(dogImgH, dogImgW, (float) 0);
  for (int j=0; j<edgeY.cols; j++){
    int x = (int) floor(edgeX(0,j));
    int y = (int) std::max((double) 0, floor(edgeY(0,j)));
    //printf("j: %d\tx: %d\ty: %d\n", j, x, y);
    float gradX = dogX(0,j);
    float gradY = dogY(0,j);
    //dogImg(y,x) = gradX*gradX + gradY*gradY;
    dogImg(y,x) = gradX + gradY;
  }

  // visualise it
    dogImg = cv::abs(dogImg);
    double min, max;
    cv::minMaxLoc(dogImg, &min, &max);
    //printf("min: %f\tmax: %f\n", min, max);
    dogImg /= max;
    dogImg *= 255;
    cv::Mat_<uint8_t> dogImgU;
    dogImg.convertTo(dogImgU, CV_8UC1);
    cv::circle(dogImgU, cv::Point2i(edgeX(c),edgeY(c)), 5, 128, 1, 8);

    // draw local extrema
    cv::imshow("dogImg", dogImgU);
    cv::waitKey(0);
}


// Computes a gradient orientation histogram at a specified pixel
static float calcOrientationHist( const cv::Mat& imgX, const cv::Mat& imgY,
    cv::Point pt, int radius, float sigma, float* hist, int n ){
  // sigma = 1.5 * scale octave
  int i, j, k, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);
    cv::AutoBuffer<float> buf(len*4 + n+4);
    //float *X = buf.data(), *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float *X = &buf[0], *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float* temphist = W + len + 2;

    for( i = 0; i < n; i++ ){
      temphist[i] = 0.f;
    }

    //for( i = -radius, k = 0; i <= radius; i++ ){
    //for( i = 0, k = 0; i <= 0; i++ ){
    k = 0;
        for( j = -radius; j <= radius; j++ ){
            int x = pt.x + j; int y = pt.y + j;
            if( x <= 0 || x >= imgX.cols - 1 ){ continue; }
            if( y <= 0 || y >= imgY.cols - 1 ){ continue; }
            //float dx = (float)(img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y, x-1));
            //float dy = (float)(img.at<sift_wt>(y-1, x) - img.at<sift_wt>(y+1, x));
            float dx = (float)(imgX.at<sift_wt>(0, x+1) - imgX.at<sift_wt>(0, x-1));
            float dy = (float)(imgY.at<sift_wt>(0, y-1) - imgY.at<sift_wt>(0, y+1));
           
            // Section 5 (below the eq)
            // W: each sample added to the histogra, is weighted by its
            // gradient magnitude and by a gaussian weighted circular window
            // with a sigma that 1.5 times og the scale of keypoint
            X[k] = dx; Y[k] = dy; 
            W[k] = (j*j + j*j)*expf_scale; // TODO: very rough approaximation of the distance, improve it 
            //W[k] = (dx*dx + dy*dy)*expf_scale; 
            k++;
        }
    //}

    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true); // The angle is measured in degrees and varies from 0 to 360 degrees.
    cv::hal::magnitude32f(X, Y, Mag, len);

    k = 0;
#if CV_AVX2
    if( USE_AVX2 )
    {
        __m256 __nd360 = _mm256_set1_ps(n/360.f);
        __m256i __n = _mm256_set1_epi32(n);
        int CV_DECL_ALIGNED(32) bin_buf[8];
        float CV_DECL_ALIGNED(32) w_mul_mag_buf[8];
        for ( ; k <= len - 8; k+=8 )
        {
            __m256i __bin = _mm256_cvtps_epi32(_mm256_mul_ps(__nd360, _mm256_loadu_ps(&Ori[k])));

            __bin = _mm256_sub_epi32(__bin, _mm256_andnot_si256(_mm256_cmpgt_epi32(__n, __bin), __n));
            __bin = _mm256_add_epi32(__bin, _mm256_and_si256(__n, _mm256_cmpgt_epi32(_mm256_setzero_si256(), __bin)));

            __m256 __w_mul_mag = _mm256_mul_ps(_mm256_loadu_ps(&W[k]), _mm256_loadu_ps(&Mag[k]));

            _mm256_store_si256((__m256i *) bin_buf, __bin);
            _mm256_store_ps(w_mul_mag_buf, __w_mul_mag);

            temphist[bin_buf[0]] += w_mul_mag_buf[0];
            temphist[bin_buf[1]] += w_mul_mag_buf[1];
            temphist[bin_buf[2]] += w_mul_mag_buf[2];
            temphist[bin_buf[3]] += w_mul_mag_buf[3];
            temphist[bin_buf[4]] += w_mul_mag_buf[4];
            temphist[bin_buf[5]] += w_mul_mag_buf[5];
            temphist[bin_buf[6]] += w_mul_mag_buf[6];
            temphist[bin_buf[7]] += w_mul_mag_buf[7];
        }
    }
#endif
    for( ; k < len; k++ )
    {
        int bin = cvRound((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }

    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];

    i = 0;
#if CV_AVX2
    if( USE_AVX2 )
    {
        __m256 __d_1_16 = _mm256_set1_ps(1.f/16.f);
        __m256 __d_4_16 = _mm256_set1_ps(4.f/16.f);
        __m256 __d_6_16 = _mm256_set1_ps(6.f/16.f);
        for( ; i <= n - 8; i+=8 )
        {
#if CV_FMA3
            __m256 __hist = _mm256_fmadd_ps(
                _mm256_add_ps(_mm256_loadu_ps(&temphist[i-2]), _mm256_loadu_ps(&temphist[i+2])),
                __d_1_16,
                _mm256_fmadd_ps(
                    _mm256_add_ps(_mm256_loadu_ps(&temphist[i-1]), _mm256_loadu_ps(&temphist[i+1])),
                    __d_4_16,
                    _mm256_mul_ps(_mm256_loadu_ps(&temphist[i]), __d_6_16)));
#else
            __m256 __hist = _mm256_add_ps(
                _mm256_mul_ps(
                        _mm256_add_ps(_mm256_loadu_ps(&temphist[i-2]), _mm256_loadu_ps(&temphist[i+2])),
                        __d_1_16),
                _mm256_add_ps(
                    _mm256_mul_ps(
                        _mm256_add_ps(_mm256_loadu_ps(&temphist[i-1]), _mm256_loadu_ps(&temphist[i+1])),
                        __d_4_16),
                    _mm256_mul_ps(_mm256_loadu_ps(&temphist[i]), __d_6_16)));
#endif
            _mm256_storeu_ps(&hist[i], __hist);
        }
    }
#endif
    for( ; i < n; i++ )
    {
    // smooth the histogram
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    return maxval;
}


//
// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
static bool adjustLocalExtrema( const std::vector<cv::Mat>& dog_pyr, cv::KeyPoint& kpt, int octv,
                                int& layer, int& r, int& c, int nOctaveLayers,
                                float contrastThreshold, float edgeThreshold, float sigma )
{
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE); // pixel type size (1 or 48)
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;

    for( ; i < SIFT_MAX_INTERP_STEPS; i++ ) // <5
    {
        int idx = octv*(nOctaveLayers+2) + layer; // o*(nOctaveLayers+2)+layer
        const cv::Mat& img = dog_pyr[idx];
        const cv::Mat& prev = dog_pyr[idx-1];
        const cv::Mat& next = dog_pyr[idx+1];

        cv::Vec2f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                 //(img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                 (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

        float v2 = (float)img.at<sift_wt>(r, c)*2;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        //float dyy = 0; //(img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
        //float dxy = 0; //(img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     //img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
                     prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
        //float dys = 0; //(next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
                     //prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;

        //cv::Matx33f H(dxx, dxy, dxs,
        //          dxy, dyy, dys,
        //          dxs, dys, dss);
        cv::Matx22f H(dxx, dxs,
                      dxs, dss);

        cv::Vec2f X = H.solve(dD, cv::DECOMP_LU);

        xi = -X[1];
        //xr = -X[1];
        xc = -X[0];
        
        // if the offset is larger than 0.5 in any dimension, then it means
        // that the extremum lies closer to a different sample point.
        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f ){
            //printf("That the extremum lies closer to a different sample point.\n");
            break;
        }

        if( std::abs(xi) > (float)(INT_MAX/3) ||
            std::abs(xr) > (float)(INT_MAX/3) ||
            std::abs(xc) > (float)(INT_MAX/3) )
            return false;
        

        c += cvRound(xc);
        //r += cvRound(xr);
        layer += cvRound(xi);

        if( layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER){
            //r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
            return false;
        }
    }

    // ensure convergence of interpolation
    if( i >= SIFT_MAX_INTERP_STEPS )
        return false;
    
    
    // eliminating edge responses
    if(0){
      int idx = octv*(nOctaveLayers+2) + layer;
      const cv::Mat& img = dog_pyr[idx];
      const cv::Mat& prev = dog_pyr[idx-1];
      const cv::Mat& next = dog_pyr[idx+1];
      cv::Matx21f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
          //0, //(img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
          (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
      //float t = dD.dot(cv::Matx31f(xc, xr, xi));
      float t = dD.dot(cv::Matx21f(xc, xi));

      contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f; // taylor expansion to get the contrast value 
      // at the interpolated extrema position. But since it is a taylor
      // expansion of order 1, I don't understand why there is 0.5
      if( std::abs( contr ) * nOctaveLayers < contrastThreshold ){
        return false;
      }

      // principal curvatures are computed using the trace and det of Hessian
      float v2 = img.at<sift_wt>(r, c)*2.f;
      float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
      float dyy = 0; //(img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
      float dxy = 0; //(img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
      //img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
      float tr = dxx + dyy;
      float det = dxx * dyy - dxy * dxy;

      if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det ){
        return false;
      }
    }

    // (x,y) in image of size the images in pyr[0,:]
    // so if you doubled the image size (as is done by default), these are the
    // coordinates in an image with size 2* the original image size.
    kpt.pt.x = (c + xc) * (1 << octv); // 1*pow(2,octv)
    //std::cout << kpt.pt.x << std::endl;
    kpt.pt.y = 0; //(r + xr) * (1 << octv);
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16); // layer shifted by 8 bits i.e. layer*pow(2,8)
    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2; // ~sigma*pow(2,idx) * pow(2,octv) * 2
    kpt.response = std::abs(contr);

    return true;
}


class findScaleSpaceExtremaComputer : public cv::ParallelLoopBody{
public:
    findScaleSpaceExtremaComputer(
        int _o,
        int _i,
        int _threshold,
        int _idx,
        int _step,
        int _cols,
        int _nOctaveLayers,
        double _contrastThreshold,
        double _edgeThreshold,
        double _sigma,
        const std::vector<cv::Mat>& _gauss_pyrX,
        const std::vector<cv::Mat>& _gauss_pyrY,
        const std::vector<cv::Mat>& _dog_pyrX,
        const std::vector<cv::Mat>& _dog_pyrY,
        const std::vector<cv::Mat>& _dog_pyr,
        cv::TLSData<std::vector<cv::KeyPoint> > &_tls_kpts_struct)

        : o(_o),
          i(_i),
          threshold(_threshold),
          idx(_idx),
          step(_step),
          cols(_cols),
          nOctaveLayers(_nOctaveLayers),
          contrastThreshold(_contrastThreshold),
          edgeThreshold(_edgeThreshold),
          sigma(_sigma),
          gauss_pyrX(_gauss_pyrX),
          gauss_pyrY(_gauss_pyrY),
          dog_pyrX(_dog_pyrX),
          dog_pyrY(_dog_pyrY),
          dog_pyr(_dog_pyr),
          tls_kpts_struct(_tls_kpts_struct) { }
    void operator()( const cv::Range& range ) const override{
      static const int n = SIFT_ORI_HIST_BINS;
      float hist[n];

      const cv::Mat& img = dog_pyr[idx];
      const cv::Mat& prev = dog_pyr[idx-1];
      const cv::Mat& next = dog_pyr[idx+1];

      std::vector<cv::KeyPoint> *tls_kpts = tls_kpts_struct.get();

      cv::KeyPoint kpt;
      int r = 0;
      const sift_wt* currptr = img.ptr<sift_wt>(r);
      const sift_wt* prevptr = prev.ptr<sift_wt>(r);
      const sift_wt* nextptr = next.ptr<sift_wt>(r);

      for( int c = SIFT_IMG_BORDER; c < cols-SIFT_IMG_BORDER; c++){
        sift_wt val = currptr[c];
        if( std::abs(val) > threshold && 
            // local maxima over 8 neighbours
            ((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
              val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
              val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1]) ||
             // local minima over 8 neighbours
             (val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
              val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
              val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1]) //|| 
            ) 
          ){

          int r1 = r, c1 = c, layer = i;
          // refine the maxima position with subpixel precision
          if( !adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
                nOctaveLayers, (float)contrastThreshold,
                (float)edgeThreshold, (float)sigma) ){
            continue;
          }
          float scl_octv = kpt.size*0.5f/(1 << o); // sigma*pow(2, r1/nOctaveLayers) (trust me)
          float omax = calcOrientationHist(
              gauss_pyrX[o*(nOctaveLayers+3) + layer],
              gauss_pyrY[o*(nOctaveLayers+3) + layer],
              cv::Point(c1, r1),
              cvRound(SIFT_ORI_RADIUS * scl_octv), // 4.5*scal_octv
              SIFT_ORI_SIG_FCTR * scl_octv,
              hist, n);
          float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO); // 80% of highest peak
          // Any other local peak within 80% of the highest peak is
          // used to also create a keypoint with that orientation
          // i.e. create multiple keypoints at the same location and scale
          // with multiple orientation.
          for( int j = 0; j < n; j++ ){ // n=nbins
            int l = j > 0 ? j - 1 : n - 1; // left bin
            int r2 = j < n-1 ? j + 1 : 0; // right bin

            // if the the j-th bin is higher than its neighbour and
            // higher than the threshold
            if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr ){
              // interpolate the bin abscisse i.e. compute barycentre
              // based on the bin magniture to get the new angle
              float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
              bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
              // is this the peak interpolation for better accuracy ?
              // (me one year later: yes)
              kpt.angle = 360.f - (float)((360.f/n) * bin); 
              if(std::abs(kpt.angle - 360.f) < FLT_EPSILON){
                kpt.angle = 0.f;
              }
              tls_kpts->push_back(kpt);
              break; // TODO: I want to keep only one orientation for now
            }
          }
        }
      }
    }
private:
    int o, i;
    int threshold;
    int idx, step, cols;
    int nOctaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;
    const std::vector<cv::Mat>& gauss_pyrX;
    const std::vector<cv::Mat>& gauss_pyrY;
    const std::vector<cv::Mat>& dog_pyrX;
    const std::vector<cv::Mat>& dog_pyrY;
    const std::vector<cv::Mat>& dog_pyr;
    cv::TLSData<std::vector<cv::KeyPoint> > &tls_kpts_struct;
};


// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
void SIFT::findScaleSpaceExtrema(
    const std::vector<cv::Mat>& gauss_pyrX,  
    const std::vector<cv::Mat>& gauss_pyrY, 
    const std::vector<cv::Mat>& dog_pyrX,
    const std::vector<cv::Mat>& dog_pyrY, 
    const std::vector<cv::Mat>& dog_pyr,
    std::vector<cv::KeyPoint>& keypoints){

  // TODO: adjust this value if needed
  const int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
  //printf("findScaleSpaceExtrema::threshold: %d\n", threshold);

  keypoints.clear();
  cv::TLSData<std::vector<cv::KeyPoint> > tls_kpts_struct;

  for( int o = 0; o < nOctaves; o++ ){ // 0->6
    for( int i = 1; i <= nOctaveLayers; i++ ){ // 1->3
      const int idx = o*(nOctaveLayers+2)+i;
      const cv::Mat& img = dog_pyr[idx];
      const int step = 0; //(int)img.step1(); // TODO: delete, we are not using it as I am in 1D
      const int cols = img.cols;

      //parallel_for_(cv::Range(SIFT_IMG_BORDER, rows-SIFT_IMG_BORDER),
      // TODO: update parellel loop to split columns
      parallel_for_(cv::Range(0, 1),
          findScaleSpaceExtremaComputer(
            o, i, threshold, idx, step, cols,
            nOctaveLayers,
            contrastThreshold,
            edgeThreshold,
            sigma,
            gauss_pyrX, gauss_pyrY,
            dog_pyrX, dog_pyrY,
            dog_pyr,
            tls_kpts_struct));
    }
  }

  std::vector<std::vector<cv::KeyPoint>*> kpt_vecs;
  tls_kpts_struct.gather(kpt_vecs);
  for (size_t i = 0; i < kpt_vecs.size(); ++i) {
    keypoints.insert(keypoints.end(), kpt_vecs[i]->begin(), kpt_vecs[i]->end());
  }
}


void SIFT::detectAndCompute(cv::Mat_<float>& edge, std::vector<cv::KeyPoint>&
    keypoints, int nfeatures, std::string debug_prefix){

  // denoise edge
  int denoise_window = 5; // TODO: play with me
  cv::Mat_<float> edgeX = edge.row(0).clone();
  cv::Mat_<float> edgeY = edge.row(1).clone();
  cv::blur(edgeX, edgeX, cv::Size(denoise_window, denoise_window));
  cv::blur(edgeY, edgeY, cv::Size(denoise_window, denoise_window));

  // create init edge
  cv::Mat_<float> baseX, baseY;
  createInitialEdge(edgeX, firstOctave < 0, baseX);
  createInitialEdge(edgeY, firstOctave < 0, baseY);
  
  // get initial edge length
  cv::Mat_<float> base(2, baseX.cols, (float) 0);
  baseX.copyTo(base.row(0));
  baseY.copyTo(base.row(1));
  double length = cv::norm(base.colRange(0, base.cols-1) - base.colRange(1, base.cols));
  
  // deduce number of octave from edge length
  nOctaves = cvRound(std::log(length) / std::log(2.) - 2) - firstOctave;

  // build gaussian pyramids
  std::vector<cv::Mat> gauss_pyrX, dog_pyrX;
  std::vector<cv::Mat> gauss_pyrY, dog_pyrY;

  buildGaussianPyramid(baseX, gauss_pyrX); // \nabla_x f(s)
  buildGaussianPyramid(baseY, gauss_pyrY); // \nabla_y f(s)
  //debugGaussianPyramid(gauss_pyrX, gauss_pyrY, debug_prefix);
  
  buildDoGPyramid(gauss_pyrX, dog_pyrX); // \Delta_x f(s) 
  buildDoGPyramid(gauss_pyrY, dog_pyrY); // \Delta_y f(s)
  //debugDoGPyramid(gauss_pyrX, gauss_pyrY, dog_pyrX, dog_pyrY, nOctaves, nOctaveLayers, debug_prefix);
  
  std::vector<cv::Mat> dogpyr; // \Delta f (I think this is what they call curvature)
  dogpyr.resize( nOctaves*(nOctaveLayers + 2) );
  for(int a=0; a<nOctaves*(nOctaveLayers+2); a++){
    int o = a / (nOctaveLayers + 2);
    int i = a % (nOctaveLayers + 2);
    const cv::Mat& src1 = dog_pyrX[o*(nOctaveLayers + 2) + i];
    const cv::Mat& src2 = dog_pyrY[o*(nOctaveLayers + 2) + i];

    //dogpyr[o*(nOctaveLayers + 2) + i] = src1 + src2; // ~laplacian

    // acceleration square L2 norm
    cv::pow(src1, 2, src1);
    cv::pow(src2, 2, src2);
    dogpyr[o*(nOctaveLayers + 2) + i] = src1 + src2; // ~laplacian

    // acceleration L1 norm
    //dogpyr[o*(nOctaveLayers + 2) + i] = cv::abs(src1) + cv::abs(src2); // ~laplacian

    //// acceleration L\infty norm
    //cv::absdiff(src1, 0, src1);
    //cv::absdiff(src2, 0, src2);
    //cv::max(src1, src2, dogpyr[o*(nOctaveLayers + 2) + i]);
  }
  //printf("Laplacian approx OK\n");
  
  //std::vector<cv::KeyPoint> kp;
  findScaleSpaceExtrema(gauss_pyrX, gauss_pyrY, dog_pyrX, dog_pyrY, dogpyr, keypoints);
  cv::KeyPointsFilter::removeDuplicated(keypoints);
  //if( nfeatures > 0 ){
  //  //for(size_t i=0; i<keypoints.size(); i++){
  //  //  printf("%f\n", keypoints[i].response);
  //  //}
  //  cv::KeyPointsFilter::retainBest(keypoints, nfeatures);
  //  //printf("# kp after filtering: %lu\n", keypoints.size());
  //}
  
  if( firstOctave < 0 ){
    for( size_t i = 0; i < keypoints.size(); i++ ) {
      cv::KeyPoint& kpt = keypoints[i];
      float scale = 1.f/(float)(1 << -firstOctave);
      kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
      kpt.pt *= scale;
      kpt.size *= scale;
    }
  }
}

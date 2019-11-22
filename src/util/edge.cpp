/**
 * \file shape_context/src/edge.cpp
 * \brief Set of primitives to process semantic maps and colors.
 */
#include "stdio.h"
#include "stdlib.h"
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "util/edge.h"
#include "util/semantic.h"
#include "util/types.h"
#include "util/cst.h"

void edge2mat(std::vector<cv::Point2i>& edgeV, cv::Mat_<float>& edgeM){
  edgeM = cv::Mat_<float>(2, edgeV.size(), (float) 0);
  for(size_t i=0; i<edgeV.size(); i++){
    edgeM(0,i) = edgeV[i].x;
    edgeM(1,i) = edgeV[i].y;
  }
}

void edge2img(cv::Mat_<float>& edgeX, cv::Mat_<float>& edgeY, cv::Mat_<uint8_t>& img){
  //printf("img.rows: %d\timg.cols: %d\n", img.rows, img.cols);
  for(int i=0; i<edgeX.cols; i++){
    //printf("i: %d\n",i);
    //printf("x, y: %d\t%d\n", (int) edgeY(i), (int) edgeX(i));
    img((int)edgeY(i), (int)edgeX(i)) = 255;
  }
}

void edge2img(cv::Mat& edgeX, cv::Mat& edgeY, cv::Mat_<uint8_t>& img){
  //printf("img.rows: %d\timg.cols: %d\n", img.rows, img.cols);
  for(int i=0; i<edgeX.cols; i++){
    //printf("i: %d\n",i);
    //printf("i: %d\tx: %d\ty: %d\n", i, (int)edgeY.at<float>(i), (int)edgeX.at<float>(i));
    img((int)edgeY.at<float>(i), (int)edgeX.at<float>(i)) = 255;
  }
}

void edge2img(cv::Mat_<float>& edge, cv::Mat_<uint8_t>& img){
  cv::Mat_<float> edgeX = edge.row(0);
  cv::Mat_<float> edgeY = edge.row(1);
  edge2img(edgeX, edgeY, img);
}


void resizeEdge(cv::Mat_<float>& edgeIn, float factor, cv::Mat_<float>& edgeOut, int mode){
  //edgeOut = edgeIn.clone();
  //edgeOut *= factor;
  cv::resize(edgeIn, edgeOut, cv::Size(), factor, 1, mode);
  //printf("edgeIn.rows: %d\tedgeIn.cols: %d\n", edgeIn.rows, edgeIn.cols);
  //printf("edgeOut.rows: %d\tedgeOut.cols: %d\n", edgeOut.rows, edgeOut.cols);
}

// TODO: there probably an API that would avoid redundancy
void resizeEdge(const cv::Mat& edgeIn, float factor, cv::Mat& edgeOut, int mode){
  //edgeOut = edgeIn.clone();
  //edgeOut *= factor;
  cv::resize(edgeIn, edgeOut, cv::Size(), factor, 1, mode);
}


void contour2edges(EdgeParams params, std::vector<cv::Point2i>& contour,
    cv::Point2i imgShape, std::vector<std::vector<cv::Point2i> >& edges){
  int cols = imgShape.x;
  int rows = imgShape.y;

  //printf("contourSize: %lu\n", contour.size());
  // start at a border
  int borderStart = 0;
  for (size_t j=0; j<contour.size(); j++){
    cv::Point2i p = contour[j];
    if ((p.x == (cols-1)) || (p.x == 0) || 
        (p.y == (rows-1)) || (p.y == 0)){ // if img border
      borderStart = j;
      break;
    }
  }

  std::vector<cv::Point2i> tmp;
  // follow border until you find an edge
  for (size_t j=borderStart; j<borderStart + contour.size(); j++){
    int jj = j%contour.size();
    cv::Point2i p = contour[jj];
    if ((p.x == (cols-1)) || (p.x == 0) || 
        (p.y == (rows-1)) || (p.y == 0)){ // if img border
      if (tmp.size() == 0){ // we don't have an edge yet
        continue; 
      }
      else{ // we were having an edge and now we find a border again
        if (tmp.size() >= (size_t) params.minContourSize){
          edges.push_back(tmp); // save current edge and start again when you leave the border
        }
        tmp.clear();
      }
    }
    else{
      tmp.push_back(p);
    }
  }
  // save the last edge
  if (tmp.size() >= (size_t) params.minContourSize){
    edges.push_back(tmp);
  }
  tmp.clear();
  
//#define LOCAL_DEBUG
#ifdef LOCAL_DEBUG
  for(size_t i=0; i<edges.size(); i++){
    printf("i: %lu\n", i);
    cv::Mat_<uint8_t> mask(rows, cols, (uint8_t) 0);
    for(size_t j=0; j<edges[i].size(); j++){
      cv::Point2i p = edges[i][j];
      mask(p.y, p.x) = 255;
    }
    cv::imshow("continuous_edge", mask);
    cv::waitKey(0);
  }
#endif
}


void extractSemanticEdge(EdgeParams params, cv::Mat_<cv::Vec3b>& semImg, cv::Mat_<cv::Vec3b>& colors,
    std::map<int, std::vector<Edge> >& semanticEdges){
  // convert color semantic map to label map
  //cv::Mat_<cv::Vec3b> colors;
  //loadColors(colors);
  //
  cv::Mat_<uint8_t> labMap(semImg.rows, semImg.cols, (uint8_t) 0);
  col2lab(semImg, colors, labMap);
#ifdef LOCAL_DEBUG
    cv::Mat_<cv::Vec3b> colMap(semImg.rows, semImg.cols, cv::Vec3b(0,0,0));
    lab2col(labMap, colors, colMap);
    cv::imshow("colMap", colMap);
    cv::imshow("semImg", semImg);
#endif

  // get present label
  std::vector<int> labels(LABEL_NUM, (int) 0);
  for (int i=0; i<labMap.rows; i++){
    for (int j=0; j<labMap.cols; j++){
      labels[labMap(i,j)] = 1;
    }
  }


  // get semantic contour
  for (int label=0; label<LABEL_NUM; label++){
    if (labels[label] == 0){ 
      continue; // label i is not present in this img
    }

    // gen mask
    cv::Mat_<uint8_t> mask(labMap.rows, labMap.cols, (uint8_t) 0);
    for (int i=0; i<labMap.rows; i++){
      for (int j=0; j<labMap.cols; j++){
        if (labMap(i,j) == label){
          mask(i,j) = 255;
        }
      }
    }
#ifdef LOCAL_DEBUG
    cv::imshow("mask", mask);
    cv::waitKey(0);
#endif


    // extract contour
    std::vector<std::vector<cv::Point2i> > contoursAll;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contoursAll, hierarchy, cv::RETR_LIST,
        CV_CHAIN_APPROX_NONE);
#ifdef LOCAL_DEBUG
      cv::Mat_<uint8_t> contoursImg(mask.rows, mask.cols, (uint8_t) 0);
      cv::drawContours(contoursImg, contoursAll, -1, cv::Scalar(255),
                  1, cv::LINE_AA, hierarchy, 3);
      cv::imshow("contoursImg", contoursImg);
      cv::waitKey(0);
#endif


    // keep only long-enough contours
    std::vector<std::vector<cv::Point2i> > contoursLong;
    for (size_t i=0; i<contoursAll.size(); i++){
      //printf("contourSize: %lu\n", contoursAll[i].size());
      if (contoursAll[i].size() > (size_t) params.minContourSize){
        contoursLong.push_back(contoursAll[i]);
      }
    }
#ifdef LOCAL_DEBUG
    cv::Mat_<uint8_t> contoursLongImg(mask.rows, mask.cols, (uint8_t) 0);
    cv::drawContours(contoursLongImg, contoursLong, -1, cv::Scalar(255),
        1, cv::LINE_AA, cv::noArray(), 3);
    cv::imshow("contoursLongImg", contoursLongImg);
    cv::waitKey(0);
#endif

    // extract ~continuous components
    std::vector<Edge> edges;
    cv::Point2i imgShape(labMap.cols, labMap.rows);
    //printf("label: %d\tcontoursLong.size: %lu\n", label, contoursLong.size());
    for (size_t i=0; i<contoursLong.size(); i++){
      contour2edges(params, contoursLong[i], imgShape, edges);
#ifdef LOCAL_DEBUG
      cv::Mat_<uint8_t> contoursLongImg(mask.rows, mask.cols, (uint8_t) 0);
      cv::drawContours(contoursLongImg, contoursLong, i, cv::Scalar(255),
                  1, cv::LINE_AA, cv::noArray(), 3);
      cv::imshow("i-th contours", contoursLongImg);
      cv::waitKey(0);
#endif
    }
    //printf("# of edges: %lu\n", edges.size());
    if(edges.size()!=0){
      semanticEdges[label] = edges;
    }
  }
}


void subsampleContour(std::vector<cv::Point2i>& contour, int step, std::vector<cv::Point2f>& subContour){
  // TODO: find a smarter subsample
  for (size_t i=0; i<contour.size(); i+=step){
    subContour.push_back(cv::Point2f(contour[i].x, contour[i].y));
  }
  // add last point if not already added
  cv::Point2i p = contour[contour.size() - 1];
  cv::Point2f pf (p.x, p.y);
  if (subContour[subContour.size()-1] != pf){
    subContour.push_back(pf);
  }
}

void kpIdx2kp(cv::Mat_<float>& edge, std::vector<cv::KeyPoint>& kpsIdx, std::vector<cv::KeyPoint>& kps){
  for(size_t i=0; i<kpsIdx.size(); i++){
    float idx = kpsIdx[i].pt.x;
    //std::cout << kpsIdx[i].pt.x << " " << idx << std::endl;
    float idx_prev = floor(idx);
    float idx_next = ceil(idx);
    //printf("idx: %f\tidx_prev: %f\tidx_next: %f\n", idx, idx_prev, idx_next);

    //float w_prev = 1. - idx - idx_prev;
    //float w_next = 1. - idx_next - idx;

    float w_prev = 1. - (idx - idx_prev);
    float w_next = 1. - (idx_next - idx);
    //printf("w_prev: %f\tw_next: %f\n", w_prev, w_next);
    
    //// the kp coordinate are the middle of the enclosing edge points
    //float x = (edge(0, idx_prev) + edge(0, idx_next))/2;
    //float y = (edge(1, idx_prev) + edge(1, idx_next))/2;

    // the kp coordinate is the barycenter of the enclosing edge points
    float x = w_prev*edge(0, (int) idx_prev) + w_next*edge(0, (int) idx_next);
    float y = w_prev*edge(1, (int) idx_prev) + w_next*edge(1, (int) idx_next);
    x /= (w_prev + w_next);
    y /= (w_prev + w_next);
    //std::cout << w_prev + w_next << std::endl; // check sum of barycenter coeffs is 1
    //printf("left pt (%f, %f)\tpt (%f, %f)\tright pt (%f, %f)\n", 
    //    edge(0, (int) idx_prev), edge(1, (int) idx_prev),
    //    x, y,
    //    edge(0, (int) idx_next), edge(1, (int) idx_next));

    cv::KeyPoint kp = kpsIdx[i];
    kp.pt.x = x;
    kp.pt.y = y;
    kps.push_back(kp);
  }
}

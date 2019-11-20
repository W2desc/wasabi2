/**
 * \file examples/cmu.cpp
 * \brief Extracts local features from the database images to file.
 */

#include "stdio.h"
#include "stdlib.h"
#include <pthread.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "dataset/survey.h"
#include "feature/sift.h"
#include "feature/scd.h"
#include "util/edge.h"
#include "util/semantic.h"
#include "util/io.h"
#include "util/types.h"
#include "util/cst.h"

#define DATA_DIR "meta/cmu/bgr/"
#define SEG_DIR "meta/cmu/seg/"

#define MIN_CONTOUR_SIZE 500
#define MIN_EDGE_SIZE 400

typedef std::map<int, std::vector<Edge> > Label2edges;
    
static const unsigned int num_threads = 8;

struct LocalFeatures{
  pthread_mutex_t mutex;
  unsigned int index;

  std::vector<Edge> edges;

  std::vector<int> edgeLabel;

  std::vector<std::vector<cv::KeyPoint> > kps;

  std::vector<cv::Mat_<float> > des;

  LocalFeatures(const std::vector<Edge>& edges, const std::vector<int>& edgeLabel): 
    edges(edges), 
    edgeLabel(edgeLabel){
    pthread_mutex_init(&mutex, NULL);
    index = 0;
    kps.resize(edges.size());
    des.resize(edges.size());
  }
};


void* feature_thread(void* arg){
  LocalFeatures* lf = (LocalFeatures*) arg;

  // init detector
  int firstOctave = -1;
  double sigma = 1.6;
  int nOctaveLayers = 3;
  SIFT detector(firstOctave, sigma, nOctaveLayers);

  // init descriptor
  int nAngularBins = 12;
  int nRadialBins = 4;
  // kp specific params, will be updated for each kp
  float innerRadius = 0.2;
  float outerRadius = 10;
  bool rotationInvariant = false;
  SCD scd(nAngularBins, nRadialBins, innerRadius, outerRadius, rotationInvariant);

  while(1){
    unsigned int i=0;
    pthread_mutex_lock(&(lf->mutex));
    i = lf->index;
    lf->index += 1;
    pthread_mutex_unlock(&(lf->mutex));

    if (i>=lf->edges.size()){
      break; // all edges have been processed
    }

    // get edge
    Edge edge = lf->edges[i];
    cv::Mat_<float> edgeM;
    edge2mat(edge, edgeM); // std::vector -> cv::Mat

    // detect       
    std::vector<cv::KeyPoint> kpIdx, kp;
    detector.detectAndCompute(edgeM, kpIdx, -1, "");
    if (kpIdx.size() == 0){
      continue;
    }
    kpIdx2kp(edgeM, kpIdx, kp);

    // describe
    cv::Mat_<float> des;
    scd.describe(edgeM, kpIdx, des);
    
    // store
    pthread_mutex_lock(&(lf->mutex));
    lf->kps[i] = kp;
    lf->des[i] = des.clone();
    pthread_mutex_unlock(&(lf->mutex));
  }

  return NULL;
}


int main(int argc, char* argv[]){
  if (argc == 1){
    printf("Arguments:\n");
    printf("1. sliceId\n");
    printf("2. camId\n");
    printf("3. surveyId\n");
    exit(0);
  }

  if (argc != 4){
    printf("Error: wrong number of argument\n");
    printf("1. sliceId\n");
    printf("2. camId\n");
    printf("3. surveyId\n");
    exit(1);
  }

  int sliceId = atoi(argv[1]);
  int camId = atoi(argv[2]);
  int surveyId = atoi(argv[3]);

  char metaFn[256];
  if (surveyId == -1){
    sprintf(metaFn, "meta/cmu/surveys/%d/%d_c%d_db/pose.txt", sliceId, sliceId, camId);
  }
  else{
    sprintf(metaFn, "meta/cmu/surveys/%d/%d_c%d_%d/pose.txt", sliceId, sliceId, camId, surveyId);
  }
  printf("metaFn: %s\n", metaFn);


  CMUSurvey survey(metaFn, std::string(DATA_DIR), std::string(SEG_DIR));
  size_t surveySize = survey.size();
  printf("surveySize: %lu\n", surveySize);

  cv::Mat_<cv::Vec3b> colors;
  loadColors(colors);

  time_t start_time, end_time;
  time(&start_time);

  for (size_t i=0; i<survey.size(); i++){
    if (i%10==0){
      time(&end_time);
      double duration = end_time - start_time;
      printf("%lu/%lu %02d:%02d(s)\n", i, survey.size(), (int) (round(duration)/60), ((int) round(duration))%60);
    }
    std::string fn;
    survey.getImgFn(i, fn);
    imgFn2rootFn(fn,fn);

    cv::Mat_<cv::Vec3b> semImg;
    survey.getSemanticImg(i, semImg);

    EdgeParams params(MIN_CONTOUR_SIZE, MIN_EDGE_SIZE);
    Label2edges semanticEdges;
    extractSemanticEdge(params, semImg, colors, semanticEdges);

    // get labels present in img
    std::vector<Edge> edges;
    std::vector<int> edgeLabel;
    std::vector<int> labels(LABEL_NUM, 0);
    for (Label2edges::iterator it = semanticEdges.begin(); it!=semanticEdges.end(); it++){
      int label = it->first;
      labels[label] = 1;

      std::vector<Edge>& tmp = it->second;
      for (size_t j = 0; j<tmp.size(); j++){
        edges.push_back(tmp[j]);
        edgeLabel.push_back(label);
      }
    }
  
    // detect and describe local feature on edges
    LocalFeatures LF(edges, edgeLabel);
    pthread_t th[num_threads];
    for (unsigned int i=0;i<num_threads;i++) {
      pthread_create(th+i, NULL, feature_thread, &LF);
    }
    for (unsigned int i=0;i<num_threads;i++) {
      pthread_join(th[i],NULL);
    }
    
    int prev_label = -1;
    char file_mode[8];
    for (size_t j=0; j<edges.size(); j++){
      int label = edgeLabel[j];
      if (label != prev_label){
        sprintf(file_mode, "w");
        prev_label = label;
      }
      else{
        sprintf(file_mode, "a");
      }
      // write  des to file
      char outFn[512];
      if (sliceId < 10){ // not the prettiest way to do this
        sprintf(outFn, "res/cmu_urban/features/des/%d/%s.txt", label, fn.c_str());
      }
      else{
        sprintf(outFn, "res/cmu_park/features/des/%d/%s.txt", label, fn.c_str());
      }
      faiss_des2file(outFn, file_mode, LF.des[j]);

      //// write kp to file
      //sprintf(outFn, "res/cmu/features/kp/%d/%s.txt", label, fn.c_str());
      //faiss_kp2file(outFn, file_mode, LF.kps[j]);
    }
  }

  time(&end_time);
  double duration = end_time - start_time;
  printf("Total time: %02d:%02d(s)\n", (int) (round(duration)/60), ((int) round(duration))%60);

  return 0;
}

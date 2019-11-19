/**
 * \file src/util/io.cpp
 * \brief Primitives to read/write to disc.
 */
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "util/io.h"

void imgFn2rootFn(std::string imgFn, std::string& fnRoot){
    size_t found1 = imgFn.find(".");
    fnRoot = imgFn.substr(0, found1);
}


void loadImageFnList(char* fn, std::vector<std::string>& fns){
  FILE* fp = fopen(fn, "r");
  if (fp==NULL){
    printf("Error: failed to open file %s\n", fn);
    exit(1);
  }
  
  while (!feof(fp)) {
    char line[1024]="";
    if (fgets(line, 1024, fp)==NULL){
      break;
    }
    char fn[80];
    if (sscanf(line, "%s", fn) == 1){
      std::string line_str(fn);
      fns.push_back(line_str);
      //std::cout << line_str << "end" << std::endl;
    }
  }
  fclose(fp);
}


// colmap compliant
void saveAllLocalFeat(char* fn, std::vector<std::vector<cv::Point2f> >& kpAll, std::vector<cv::Mat_<float> >& desAll){
  int desDim = 128; // I hardcode it because I comply with colmap c++ for now.

  FILE* fp = fopen(fn, "w");
  if (fp == NULL){
    printf("Error: failed to open file %s\n", fn);
    exit(1);
  }

  // get number of local des
  int localDesNum = 0;
  for (size_t i=0; i<kpAll.size(); i++){
    localDesNum += (int) kpAll[i].size();
  }
  fprintf(fp, "%d %d\n", localDesNum, desDim);

  for (size_t i=0; i<kpAll.size(); i++){
    std::vector<cv::Point2f> kp = kpAll[i];
    cv::Mat_<float> des = desAll[i];
    int rows = des.rows;
    int cols = des.cols;
    for (int r=0; r<rows; r++){
      fprintf(fp, "%d %d 1 0",(int) kp[r].x, (int) kp[r].y);
      //fprintf(fp, "%d", i);
      int c = 0;
      //for (int c=0; c<cols; c++){
      for (; c<cols; c++){
        fprintf(fp, " %d", (int) des(r,c));
      }
      for (; c<desDim; c++){
        fprintf(fp, " %d", (int) 0);
      }
      fprintf(fp, "\n");
    }
  }

  fclose(fp);
}



// faiss compliant
void faiss_des2file(char* fn, char* file_mode, cv::Mat_<float>& des){
  FILE* fp = fopen(fn, file_mode);
  if (fp == NULL){
    printf("Error: failed to open file %s\n", fn);
    exit(1);
  }
  fprintf(fp, "# %d %d\n", des.rows, des.cols);
  for (int i=0; i<des.rows; i++){
    for (int j=0; j<des.cols; j++){
      fprintf(fp, "%d ", (int) des(i,j));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void faiss_kp2file(char* fn, char* file_mode, std::vector<cv::KeyPoint> kp){
  FILE* fp = fopen(fn, file_mode);
  if (fp == NULL){
    printf("Error: failed to open file %s\n", fn);
    exit(1);
  }
  fprintf(fp, "# %lu\n", kp.size());
  for (size_t i=0; i<kp.size(); i++){
    fprintf(fp, "%f %f\n", kp[i].pt.x, kp[i].pt.y);
  }
  fclose(fp);
}


void loadPairsToMatch(char* fn, std::map<std::string, std::vector<std::string> >& pairsToMatch){
  FILE* fp = fopen(fn, "r");
  if (fp==NULL){
    printf("Error: failed to open file %s\n", fn);
    exit(1);
  }
  
  while (!feof(fp)) {
    char line[1024]="";
    if (fgets(line, 1024, fp)==NULL){
      break;
    }
    char fn1[128], fn2[128];
    if (sscanf(line, "%s %s", fn1, fn2) == 2){
      pairsToMatch[std::string(fn1)].push_back(std::string(fn2));
    }
  }
  fclose(fp);
}

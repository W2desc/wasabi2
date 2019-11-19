/**
 * \file shape_context/src/semantic.cpp
 * \brief Set of primitives to process semantic maps and colors.
 */
#include "stdio.h"
#include "stdlib.h"

#include <opencv2/core.hpp>

#include "util/semantic.h"
#include "util/cst.h"

void col2lab(cv::Mat_<cv::Vec3b>& colMap, cv::Mat_<cv::Vec3b>& colors, cv::Mat_<uint8_t>& labMap){
  for (int c=0; c<colors.rows; c++){
    cv::Vec3b color = colors(c, 0);
    //std::cout << c << " " << color << std::endl;
    for (int i=0; i<colMap.rows; i++){
      for (int j=0; j<colMap.cols; j++){
        if (colMap(i,j) == color){
          labMap(i,j) = c;
        }
      }
    }
  }
}

void lab2col(cv::Mat_<uint8_t>& labMap, cv::Mat_<cv::Vec3b>& colors, cv::Mat_<cv::Vec3b>& colMap){
  for (int c=0; c<colors.rows; c++){
    cv::Vec3b color = colors(c, 0);
    for (int i=0; i<labMap.rows; i++){
      for (int j=0; j<labMap.cols; j++){
        if (labMap(i,j) == c){
          colMap(i,j) = color;
        }
      }
    }
  }
}


void loadColors(cv::Mat_<cv::Vec3b>& colors){
  int labelNum = 19;
  colors = cv::Mat_<cv::Vec3b>(labelNum, 1, cv::Vec3b(0,0,0));

  char fn[256];
  sprintf(fn, "%s/meta/palette_cityscapes.txt", PROJECT_DIR);
  FILE* fp = fopen(fn, "r");
  if (fp==NULL){
    printf("Error: failed to open file %s\n", fn);
    exit(1);
  }
  for (int i=0; i<labelNum; i++){
    float b,g,r;
    if(fscanf(fp, "%f %f %f\n", &b, &g, &r) != 3){
      printf("Failed to read line %d of %s\n", i, fn);
      exit(1);
    }
    colors(i,0) = cv::Vec3b((uint8_t) b, (uint8_t) g, (uint8_t)r);
    //printf("line: %d OK\n", i);
  }
  fclose(fp);
  //printf("loadColors OK\n");
}

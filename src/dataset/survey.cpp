/**
 *
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "dataset/survey.h"
#include "util/semantic.h"
#include "util/io.h"
#include "util/cst.h"

Survey::Survey(){
  imgDir = "";
  segDir = "";
  surveySize = 0;
  pixelBorder = 1;
}

Survey::Survey(std::string metaFn, std::string imgDir, std::string segDir){
  pixelBorder = 1;
  this->imgDir = imgDir;
  this->segDir = segDir;  
}

Survey::~Survey(){
}

size_t Survey::size(){
  return surveySize;
}

void Survey::checkIdx(size_t idx){
  assert(idx<surveySize);
}

void Survey::getImgFn(size_t idx, std::string& imgFn){
  imgFn = fns[idx];
}

void Survey::getImg(size_t idx, int proc, cv::Mat_<cv::Vec3b>& img){
  checkIdx(idx);
  char buff[512];
  sprintf(buff, "%s/%s", imgDir.c_str(), fns[idx].c_str());
  img = cv::imread(buff);
}


CMUSurvey::CMUSurvey(){
}

CMUSurvey::~CMUSurvey(){
}

CMUSurvey::CMUSurvey(std::string metaFn, std::string imgDir, std::string segDir){
  pixelBorder = 1;
  this->imgDir = imgDir;
  this->segDir = segDir;  
  
  std::vector<cv::Point3f> posesV; 
  FILE* fp = fopen(metaFn.c_str(), "r");
  if (fp==NULL){
    printf("Error: failed to open file %s\n", metaFn.c_str());
    exit(1);
  }
  
  int count = 0;
  while (!feof(fp)) {
    char line[1024]="";
    if (fgets(line, 1024, fp)==NULL){
      break;
    }
    char fn[256];
    double qw, qx, qy, qz, cx, cy, cz;
    if (sscanf(line, "%s %le %le %le %le %le %le %le", fn, &qw, &qx, &qy, &qz, &cx,
          &cy, &cz) == 8){
      fns.push_back(fn);
      posesV.push_back(cv::Point3f(cx, cy, cz));
      fn2id[fn] = count;
      count ++;
    }
  }
  fclose(fp);
  surveySize = fns.size();
}


void CMUSurvey::procImg(cv::Mat_<cv::Vec3b>& img){
  int rows = img.rows;
  int cols = img.cols;
  img = img(cv::Rect(1,1, rows-1, cols-1)).clone();
  cv::resize(img, img, cv::Size(0,0), 0.5, 0.5, cv::INTER_NEAREST);
}


void CMUSurvey::getSemanticImg(size_t idx, cv::Mat_<cv::Vec3b>& img){
  char buff[512];
  std::string fn = "";
  imgFn2rootFn(fns[idx], fn);
  sprintf(buff, "%s/%s.png", segDir.c_str(), fn.c_str());
  //printf("semFn: %s\n", buff);
  img = cv::imread(buff);
  
  // get rid on the border where label is 0 always
  int rows = img.rows;
  int cols = img.cols;
  cv::Rect a(1,1, rows-1, cols-1);
  img = img(cv::Rect(1,1, cols-1, rows-1)).clone();
}

void CMUSurvey::getSemanticImg(std::string fn, cv::Mat_<cv::Vec3b>& img){
  int id = fn2id[fn];
  getSemanticImg(id, img);
}

//////////////////////////////////
SymphonySurvey::SymphonySurvey(){
  new_w = 700;
  new_h = 480;
}


SymphonySurvey::~SymphonySurvey(){
}

SymphonySurvey::SymphonySurvey(std::string metaFn, std::string imgDir, std::string segDir, std::string maskDir){
  new_w = 700;
  new_h = 480;
  pixelBorder = 1;
  this->imgDir = imgDir;
  this->segDir = segDir;  
  this->maskDir = maskDir;
  
  std::vector<cv::Point3f> posesV; 
  FILE* fp = fopen(metaFn.c_str(), "r");
  if (fp==NULL){
    printf("Error: failed to open file %s\n", metaFn.c_str());
    exit(1);
  }
  
  int count = 0;
  while (!feof(fp)) {
    char line[1024]="";
    if (fgets(line, 1024, fp)==NULL){
      break;
    }
    char fn[256];
    double qw, qx, qy, qz, cx, cy, cz;
    if (sscanf(line, "%s %le %le %le %le %le %le %le", fn, &qw, &qx, &qy, &qz, &cx,
          &cy, &cz) == 8){
      fns.push_back(fn);
      posesV.push_back(cv::Point3f(cx, cy, cz));
      fn2id[fn] = count;
      count ++;
    }
  }
  fclose(fp);
  surveySize = fns.size();
}


void SymphonySurvey::procImg(cv::Mat_<cv::Vec3b>& img){
  img = img(cv::Rect(0,0, new_w, new_h)).clone();
}

void SymphonySurvey::cleanSemanticImg(std::string maskFn, cv::Mat_<cv::Vec3b>& semImg){
  // TODO: outsouce this
  cv::Mat_<cv::Vec3b> colors;
  loadColors(colors);

  cv::Mat_<uint8_t> mask = cv::imread(maskFn, cv::IMREAD_UNCHANGED);

  cv::Mat_<uint8_t> labMap(semImg.rows, semImg.cols, (uint8_t) 0);
  col2lab(semImg, colors, labMap);

  // vegetation denoising
  for (int i=0; i<labMap.rows; i++){
    for (int j=0; j<labMap.cols; j++){
      if ((labMap(i,j) == 4) || (labMap(i,j)==3)){
        labMap(i,j) = 8;
      }
    }
  }
  //lab2col(labMap, colors, semImg);
  //cv::imshow("debug", semImg);
  //cv::waitKey(0);

  // water denoising
  for (int i=0; i<labMap.rows; i++){
    for (int j=0; j<labMap.cols; j++){
      if (mask(i,j) == 1){
        labMap(i,j) = 255;
      }
    }
  }

  //lab2col(labMap, colors, semImg);
  //cv::imshow("debug", semImg);
  //cv::waitKey(0);


  for (int i=0; i<labMap.rows; i++){
    for (int j=0; j<labMap.cols; j++){
      if (labMap(i,j) == 0){
        labMap(i,j) = 8;
      }
    }
  }

  for (int i=0; i<labMap.rows; i++){
    for (int j=0; j<labMap.cols; j++){
      if (labMap(i,j) == 255){
        labMap(i,j) = 0;
      }
    }
  }

  lab2col(labMap, colors, semImg);
}



void SymphonySurvey::getSemanticImg(size_t idx, cv::Mat_<cv::Vec3b>& img){
  char buff[512];
  std::string fn = "";
  imgFn2rootFn(fns[idx], fn);
  sprintf(buff, "%s/%s.png", segDir.c_str(), fn.c_str());
  //printf("semFn: %s\n", buff);
  img = cv::imread(buff);
  
  // get rid on the border where label is 0 always
  img = img(cv::Rect(0,0, new_w, new_h)).clone();
}

void SymphonySurvey::getSemanticImg(std::string fn, cv::Mat_<cv::Vec3b>& img){
  int id = fn2id[fn];
  getSemanticImg(id, img);
}



//////////////////////////
//void loadSurvey(int sliceId, int camId, int surveyId, Survey& survey){
//  char metaFn[256];
//  if (surveyId == -1){
//    sprintf(metaFn, "%s/%d/%d_c%d_db/pose.txt", META_DIR, sliceId, sliceId, camId);
//  }
//  else{
//    sprintf(metaFn, "%s/%d/%d_c%d_%d/pose.txt", META_DIR, sliceId, sliceId, camId, surveyId);
//  }
//  //printf("metaFn: %s\n", metaFn);
//  survey = CMUSurvey(metaFn, std::string(DATA_DIR), std::string(SEG_DIR));
//}

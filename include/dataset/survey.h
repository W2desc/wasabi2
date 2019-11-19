/**
 *
 */

#ifndef SURVEY_H
#define SURVEY_H

#include "stdio.h"
#include "stdlib.h"
#include <map>
#include <vector>
#include <string>

#include <opencv2/core.hpp>

//struct Pose {
//  /**
//   * \var t
//   * \brief camera translation cam -> world
//   */
//  cv::Point3f t; 
//  
//  /**
//   * \var c
//   * \brief camera center
//   */
//  cv::Point3f c;
//
//  cv::Mat_<float> R(3, 3, (float) 0); // cam -> world
//  
//};


class Survey {
  protected:
    std::string imgDir;
    std::string segDir;
    std::vector<std::string> fns;
    std::vector<cv::Point3f> poses;
    
    /**
     * \param fn2id
     * \brief Map img fn to img rank in the survey.
     */
    std::map<std::string, int> fn2id;
    
    size_t surveySize;
    int pixelBorder;


  public:
    Survey();
    ~Survey();
    Survey(std::string metaFn, std::string imgDir, std::string segDir);

    size_t size();

    void getFns(std::vector<std::string>& fns);

    void getPoses(std::vector<cv::Point3f>& poses);

    void update(std::vector<int>& ok);

    void checkIdx(size_t idx);

    void getImgFn(size_t idx, std::string& imgFn);

    void getPose(size_t idx);

    virtual void procImg(cv::Mat_<cv::Vec3b>& img) = 0;

    void getImg(size_t idx, int proc, cv::Mat_<cv::Vec3b>& img);

    virtual void getSemanticImg(size_t idx, cv::Mat_<cv::Vec3b>& img) = 0;
};


class CMUSurvey: public Survey{
  public:
    CMUSurvey();
    ~CMUSurvey();
    CMUSurvey(std::string metaFn, std::string imgDir, std::string segDir);

    void procImg(cv::Mat_<cv::Vec3b>& img);
    void getSemanticImg(size_t idx, cv::Mat_<cv::Vec3b>& img);
    void getSemanticImg(std::string fn, cv::Mat_<cv::Vec3b>& img);
};

class SymphonySurvey: public Survey{
  public:
    SymphonySurvey();
    ~SymphonySurvey();
    SymphonySurvey(std::string metaFn, std::string imgDir, std::string segDir, std::string maskDir);

    void procImg(cv::Mat_<cv::Vec3b>& img);
    void getSemanticImg(size_t idx, cv::Mat_<cv::Vec3b>& img);
    void getSemanticImg(std::string fn, cv::Mat_<cv::Vec3b>& img);
    void cleanSemanticImg(std::string maskFn, cv::Mat_<cv::Vec3b>& semImg);
    

  public:
    int new_w;
    int new_h;
    std::string maskDir;
};


void loadSurvey(int sliceId, int camId, int surveyId, Survey& survey);

//class Survey{
//  public:
//    Survey();
//
//    Survey(std::string img_dir, std::string seg_dir, std::string meta_fn);
//
//    ~Survey();
//
//    void load();
//
//    int get_size();
//
//    void check_idx();
//
//    std::string get_img_fn(size_t idx);
//
//    void get_img(size_t idx, cv::Mat_<cv::Vec3b>& img);
//
//    virtual void proc_img(size_t idx, cv::Mat_<cv::Vec3b>& img) = 0;
//
//    virtual void get_semantic_img(size_t idx, cv::Mat_<cv::Vec3b>& sem_img) = 0;
//
//  private:
//    int pixel_border;
//    std::string img_dir;
//    std::string seg_dir;
//    std::string meta_fn;
//
//    std::vector<std::string> fn_v;
//    cv::Mat_<float> cam_center;
//};
//
//
//class CMUSurvey: public Survey {
//
//  void proc_img(size_t idx, cv::Mat_<cv::Vec3b>& img);
//
//  void get_semantic_img(size_t idx, cv::Mat_<cv::Vec3b>& sem_img);
//  
//
//};

#endif

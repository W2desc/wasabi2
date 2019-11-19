/**
 * \file src/util/io.h
 * \brief Primitives to read/write to disc.
 */
#ifndef IO_H
#define IO_H

#include "stdio.h"
#include "stdlib.h"
#include <string>
#include <vector>
#include <map>

#include <opencv2/core.hpp>

/**
 * \brief Remove img extension from img relative filename.
 */
void imgFn2rootFn(std::string imgFn, std::string& fnRoot);

/**
 * \brief Loads list of image relative filenames from file.
 */
void loadImageFnList(char* fn, std::vector<std::string>& fns);

/**
 * \brief Save local features of all edge to file.
 */
void saveAllLocalFeat(char* fn, std::vector<std::vector<cv::Point2f> >& kpAll, std::vector<cv::Mat_<float> >& desAll);

/**
 * \brief Loads list of image pairs to match.
 */
void loadPairsToMatch(char* fn, std::map<std::string, std::vector<std::string> >& pairsToMatch);


/**
 * \brief Writes a (N,d) matrix of N local features of dimension d to file.
 */
void faiss_des2file(char* fn, char* file_mode, cv::Mat_<float>& des);

/**
 * \brief Writes keypoint (x,y) positions of local features to file.
 */
void faiss_kp2file(char* fn, char* file_mode, std::vector<cv::KeyPoint> kp);
#endif

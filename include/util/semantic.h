/**
 * \file shape_context/include/semantic.h
 * \brief Set of primitives to process semantic maps and colors.
 */
#ifndef SEMANTIC_H
#define SEMANTIC_H

#include <opencv2/core.hpp>

void col2lab(cv::Mat_<cv::Vec3b>& colMap, cv::Mat_<cv::Vec3b>& colors, cv::Mat_<uint8_t>& labMap);

void lab2col(cv::Mat_<uint8_t>& labMap, cv::Mat_<cv::Vec3b>& colors, cv::Mat_<cv::Vec3b>& colMap);

void loadColors(cv::Mat_<cv::Vec3b>& colors);
#endif

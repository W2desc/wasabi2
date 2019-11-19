/**
 * \file shape_context/include/edge.h
 * \brief Set of primitives to process semantic maps and colors.
 */
#ifndef EDGE_H
#define EDGE_H

#include <opencv2/core.hpp>

#include "types.h"

void contour2edges(std::vector<cv::Point2i>& contour, cv::Point2i imgShape,
    std::vector<std::vector<cv::Point2i> >& edges);

void extractSemanticEdge(EdgeParams params, cv::Mat_<cv::Vec3b>& semImg, cv::Mat_<cv::Vec3b>& colors,
    std::map<int, std::vector<Edge> >& semanticEdges);

void subsampleContour(std::vector<cv::Point2i>& contour, int step, std::vector<cv::Point2f>& subContour);

void edge2mat(std::vector<cv::Point2i>& edgeV, cv::Mat_<float>& edgeM);

// TODO: replace these protottypes with template
void edge2img(cv::Mat_<float>& edgeX, cv::Mat_<float>& edgeY, cv::Mat_<uint8_t>& img);

void edge2img(cv::Mat& edgeX, cv::Mat& edgeY, cv::Mat_<uint8_t>& img);

void edge2img(cv::Mat_<float>& edge, cv::Mat_<uint8_t>& img);

void resizeEdge(cv::Mat_<float>& edgeIn, float factor, cv::Mat_<float>& edgeOut, int mode);

void resizeEdge(const cv::Mat& edgeIn, float factor, cv::Mat& edgeOut, int mode);

void kpIdx2kp(cv::Mat_<float>& edge, std::vector<cv::KeyPoint>& kpsIdx, std::vector<cv::KeyPoint>& kps);

#endif

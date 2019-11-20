/**
 * \brief Defines custom types
 */

#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include <map>

//typedef std::map<std::string, std::vector<std::string> > PairsToMatch;

//typedef int MonoLabel;
//typedef std::vector<int> MultiLabel;
typedef std::vector<cv::Point2i> Edge;
//typedef std::vector<Edge> Edges;
//typedef std::map<MonoLabel, Edges> MonoSemanticEdges;
//typedef std::map<MultiLabel, Edges> MultiSemanticEdges;
//
//typedef std::map<MonoLabel, cv::Mat_<float> > MonoSemanticDes;
//
//typedef std::vector<cv::KeyPoint> KeyPointVector;
//typedef std::vector<KeyPointVector> KeyPointVectors;
//typedef std::map<MonoLabel, KeyPointVectors> MonoSemanticKeyPoints;
//
//typedef std::map<MonoLabel, std::vector<cv::KeyPoint> > MonoSemanticKp;

// TODO: make run-time initialiser
//struct EdgeParams{
//  int minContourSize=500;
//  int minEdgeSize=400;
//};

struct EdgeParams{
  int minContourSize;
  int minEdgeSize;

  EdgeParams(){
    minContourSize = 50;
    minEdgeSize = 40;
  }

  EdgeParams(int minContourSize, int minEdgeSize): 
    minContourSize(minContourSize), 
    minEdgeSize(minEdgeSize){
  
    }
};


//typedef std::pair<int, int> MatIndex;
//typedef std::pair<MatIndex, float> MatEntry;


#endif

import os

import cv2
import numpy as np

class OpenCVFeatureExtractor(object):
    """Local feature extractor using OpenCV methods."""

    def __init__(self):
        """Instantiates a local feature extractor."""
    
    def get_local_des(self, img):
        """Detect and describe local keypoints.

        Returns:
            des_v: (N, D) np array of N local descriptors of dimension D.
        """
        _, des =self.fe.detectAndCompute(img, None)
        return des
    
    def get_local_features(self, img):
        """Detect and describe local keypoints.

        Returns:
            des_v: (N, D) np array of N local descriptors of dimension D.
        """
        kp, des = self.fe.detectAndCompute(img, None)
        return kp, des

class SIFTFeatureExtractor(OpenCVFeatureExtractor):
    """Local feature extractor using OpenCV methods."""

    def __init__(self, max_num_feat):
        if max_num_feat != -1:
            self.fe = cv2.xfeatures2d.SIFT_create(max_num_feat)
        else:
            self.fe = cv2.xfeatures2d.SIFT_create()

class SURFFeatureExtractor(OpenCVFeatureExtractor):
    """Local feature extractor using OpenCV methods."""

    def __init__(self):
        self.fe = cv2.xfeatures2d.SURF_create(400)

class ORBFeatureExtractor(OpenCVFeatureExtractor):
    """Local feature extractor using OpenCV methods."""

    def __init__(self, max_num_feat):
        self.fe = cv2.ORB_create()

class MSERFeatureExtractor(OpenCVFeatureExtractor):
    """Local feature extractor using OpenCV methods."""

    def __init__(self, max_num_feat):
        self.fe = cv2.MSER_create()

class AKAZEFeatureExtractor(OpenCVFeatureExtractor):
    """Local feature extractor using OpenCV methods."""

    def __init__(self, max_num_feat):
        self.fe = cv2.AKAZE_create()
   
class DelfFeatureExtractor(object):
    """Object to read computed delf local features from disk."""

    def __init__(self, delf_dir):
        """Instantiates a local feature extractor."""
        self.des_dir = delf_dir


    def get_local_features(self, img_fn):
        """ """

class AccFeatureExtractor(object):
    """Object to read computed delf local features from disk."""

    def __init__(self, label_num, des_dir, des_dim=48):
        """Instantiates a local feature extractor."""
        self.label_num = label_num
        self.des_dir = des_dir
        self.des_dim = 48


    def get_local_features(self, img_fn):
        """ """
        des_l = []
        for label in range(self.label_num):
            des_fn = "%s/des/%d/%s.txt"%(self.des_dir, label,
                    img_fn.split(".")[0])
            #print(des_fn)
            if os.path.exists(des_fn):
                tmp = np.loadtxt(des_fn)
                if tmp.size == 0:
                    continue
                else:
                    des_l.append(tmp)
                #print(tmp.size)
        #print(len(des_l))
        if len(des_l) == 0:
            des_v = np.zeros((1,self.des_dim))
        else:
            des_v = np.vstack(des_l)
        return des_v



class FeatureExtractorFactory(object):
    """Provides one of the registered local feature extractor."""

    def __init__(self):
        self._builders = {}
        self._builders["sift"] = SIFTFeatureExtractor
        self._builders["surf"] = SURFFeatureExtractor
        self._builders["orb"] = ORBFeatureExtractor
        self._builders["mser"] = MSERFeatureExtractor
        self._builders["akaze"] = AKAZEFeatureExtractor
        self._builders["delf"] = DelfFeatureExtractor
        self._builders["acc"] = AccFeatureExtractor

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, kwargs):
        builder = self._builders[key]
        if not builder:
            raise ValueError("Unknown feature extractor: %s"%key)
        return builder(**kwargs)

class L2Matcher(object):
    def __init__(self):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)

    def match(self, des0, des1, k=2, lowe_criterion=True):
        matches = self.matcher.knnMatch(des0, des1,k=k)
        good = []
        if lowe_criterion:
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                    good.append(m)
        else:
            good = sorted(matches, key = lambda x:x.distance)
        return good


class AkazeMatcher(object):
    def __init__(self):
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

    def match(self, des0, des1, k=2):
        return self.matcher.knnMatch(des0, des1,k=2)


class ORBMatcher(object):
    def __init__(self):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, des0, des1):
        matches = matcher.match(des0,des1)
        good = sorted(matches, key = lambda x:x.distance)
        return good


class MatcherFactory(object):
    def __init__(self):
        self._builders = {}
        self._builders["L2"] = L2Matcher
        self._builders["Akaze"] = AkazeMatcher
        self._builders["ORB"] = ORBMatcher

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, kwargs):
        builder = self._builders[key]
        if not builder:
            raise ValueError("Unknown matcher: %s"%key)
        return builder(**kwargs)


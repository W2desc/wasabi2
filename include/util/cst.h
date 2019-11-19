/**
 * \file wasabi/cpp/shape_context/cst.h
 * \brief Misc. constants.
 */

#ifndef CST_H
#define CST_H


//#define WS_DIR "/home/anonymous/ws/"
#define WS_DIR "/home/gpu_user/anonymous/ws/"
#define PROJECT_DIR WS_DIR "tf/eccv1/"

//#define SEG_DIR WS_DIR "tf/cross-season-segmentation/res/ext_cmu/"


//#define WASABI_DIR WS_DIR "tf/wasabi/"
//#define PTS_DIR WASABI_DIR "meta/edge/"
//#define DES_DIR WASABI_DIR "meta/edge/des/"
//#define IMG_SMALL_ROW 364
//#define IMG_SMALL_COL 512

#define IMG_ROW 728
#define IMG_COL 1024
#define IMG_H 768
#define IMG_W 1024

#define DATA_DIR WS_DIR "datasets/Extended-CMU-Seasons/"

#define PYDATA_DIR WS_DIR "datasets/pydata/"
//#define META_DIR PYDATA_DIR "pycmu/meta/surveys/"
#define EDGE_PTS_DIR PYDATA_DIR "pycmu/res/edge_pts/"
#define EDGE_GLOBAL_DES_DIR PYDATA_DIR "pycmu/res/edge_global_des/"
#define EDGE_LOCAL_DES_DIR PYDATA_DIR "pycmu/res/edge_local_des/"

#define LABEL_NUM 19

#define SUBSAMPLE_STEP 10

//#define HOM_DIR WASABI_DIR "cpp/shape_context/meta/hsequences/"



// LAKE MOTHERFUCKERS
#define META_DIR WS_DIR "tf/wasabi/meta/symphony/surveys/"

//// train
//#define MASK_DIR WS_DIR "datasets/lake/datasets/seg/"
//#define SEG_DIR WS_DIR "tf/cross-season-segmentation/res/cvpr_fuck/"


// qeury
#define MASK_DIR "/home/anonymous/tmp/cvpr20/mask/icra_retrieval/water/global/"
#define SEG_DIR "/home/anonymous/tmp/cvpr20/seg/icra_retrieval_OK/"

#endif

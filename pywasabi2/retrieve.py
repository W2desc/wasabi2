"""VLAD implementation.

Variation of https://github.com/jorjasso/VLAD
"""
import argparse
import os
import pickle
import time

import cv2
import numpy as np

import pywasabi2.survey
import pywasabi2.retrieval
import pywasabi2.metrics
import pywasabi2.tools_agg
import pywasabi2.local_feature

#DES_DIR = "%s/tf/eccv1/tests/p_localFeature/res/0/"%WS_DIR # cmu park
#DES_DIR = "%s/tf/eccv1/tests/p_localFeature/res/1/"%WS_DIR # cmu urban

#DES_DIR = "%s/tf/eccv1/tests/p_localFeature_lake/res/0/"%WS_DIR # lake
LABEL_NUM = 19

def gen_codebook():
    """ """
    raise NotImplementedError("TODO")


def describe_img(args, fe, centroids, img):
    """Computes a global descriptor for the image.

    Returns:
        des: (1, D) a global image descriptor of dimension D for img.
    """
    lf = fe.get_local_features(img)
    if args.agg_mode == "vlad":
        des = pywasabi2.tools_agg.lf2vlad(lf, centroids, args.vlad_norm)
    elif args.agg_mode == "bow":
        des = pywasabi2.tools_agg.lf2bow(lf, centroids, args.vlad_norm)
    else:
        raise ValueError("Unknown aggregation mode: %s"%args.agg_mode)
    return des

def get_img_des_dim(nw, dw, agg_mode):
    """Returns the global image descriptor dimension."""
    if agg_mode == "vlad":
        des_dim = nw * dw # global img descriptor dimension
    elif agg_mode == "bow":
        des_dim = nw
    else:
        raise ValueError("Unknown aggregation mode: %s"%args.agg_mode)
    return des_dim


def describe_survey(args, fe, centroids, survey):
    """ """
    NW, DW = centroids.shape[:2] # Number of Words, Dim of Words
    des_dim = get_img_des_dim(NW, DW, args.agg_mode)
    survey_size = survey.get_size()
    des_v = np.empty((survey_size, des_dim))
    for idx in range(survey.get_size()):
        if idx % 50 == 0:
            print("%d/%d"%(idx, survey.get_size()))
        img_fn = survey.get_root_img_fn(idx)
        #print(img_fn)
        des_v[idx,:] = describe_img(args, fe, centroids, img_fn)
    return des_v


def bench(args, kwargs, centroids, n_values):
    """ """
    global_start_time = time.time()

    # check if this bench already exists
    perf_dir = "res/%s/perf"%(args.data)
    mAP_fn = "%s/%d_c%d_%d_mAP.txt"%(perf_dir, args.slice_id, args.cam_id,
        args.survey_id)
    recalls_fn = "%s/%d_c%d_%d_rec.txt"%(perf_dir, args.slice_id, args.cam_id,
        args.survey_id)
    if os.path.exists(mAP_fn):
        return -1, -1 
    
    res_dir = "res/%s/retrieval/"%(args.data)

    # load db traversal
    surveyFactory = pywasabi2.survey.SurveyFactory()
    meta_fn = "%s/%d/%d_c%d_db/pose.txt"%(args.meta_dir, args.slice_id, args.slice_id, args.cam_id)
    kwargs["meta_fn"] = meta_fn
    db_survey = surveyFactory.create(args.survey_type, **kwargs)
    
    # load query traversal
    meta_fn = "%s/%d/%d_c%d_%d/pose.txt"%(args.meta_dir, args.slice_id, 
            args.slice_id, args.cam_id, args.survey_id)
    kwargs["meta_fn"] = meta_fn
    q_survey = surveyFactory.create(args.survey_type, **kwargs)

    # retrieval instance. Filters out queries without matches.
    retrieval = pywasabi2.retrieval.Retrieval(db_survey, q_survey, args.dist_pos)
    q_survey = retrieval.get_q_survey()

    # choose a local feature extractor
    feFactory = pywasabi2.local_feature.FeatureExtractorFactory()
    kwargs = {"label_num": LABEL_NUM, "des_dir": args.des_dir}
    fe = feFactory.create(args.lf_mode, kwargs)

    # describe db img
    local_start_time = time.time()
    db_des_fn = '%s/%d_c%d_db.pickle'%(res_dir, args.slice_id, args.cam_id)
    if not os.path.exists(db_des_fn): # if you did not compute the db already
        print('** Compute des for database img **')
        db_img_des_v = describe_survey(args, fe, centroids, db_survey)
        with open(db_des_fn, 'wb') as f:
            pickle.dump(db_img_des_v, f)
    else: # if you already computed it, load it from disk
        print('** Load des for database img **')
        with open(db_des_fn, 'rb') as f:
            db_img_des_v = pickle.load(f)
    duration = (time.time() - local_start_time)
    print('(END) run time: %d:%02d'%(duration/60, duration%60))


    # describe q img
    local_start_time = time.time()
    q_des_fn = '%s/%d_c%d_%d.pickle'%(res_dir, args.slice_id, args.cam_id,
            args.survey_id)
    if not os.path.exists(q_des_fn): # if you did not compute the db already
        print('\n** Compute des for query img **')
        q_img_des_v = describe_survey(args, fe, centroids, q_survey)
        with open(q_des_fn, 'wb') as f:
            pickle.dump(q_img_des_v, f)
    else: # if you already computed it, load it from disk
        print('\n** Load des for database img **')
        with open(q_des_fn, 'rb') as f:
            q_img_des_v = pickle.load(f)
    duration = (time.time() - local_start_time)
    print('(END) run time: %d:%02d'%(duration/60, duration%60))
    
    
    # retrieve each query
    print('\n** Retrieve query image **')
    local_start_time = time.time()
    d = np.linalg.norm(np.expand_dims(q_img_des_v, 1) -
            np.expand_dims(db_img_des_v, 0), ord=None, axis=2)
    order = np.argsort(d, axis=1)
    #np.savetxt(order_fn, order, fmt='%d')
    duration = (time.time() - local_start_time)
    print('(END) run time %d:%02d'%(duration/60, duration%60))

    
    # compute perf
    print('\n** Compute performance **')
    local_start_time = time.time()
    rank_l = retrieval.get_retrieval_rank(order, args.top_k)
    
    gt_name_d = retrieval.get_gt_rank("name")
    mAP = pywasabi2.metrics.mAP(rank_l, gt_name_d)

    gt_idx_l = retrieval.get_gt_rank("idx")
    recalls = pywasabi2.metrics.recallN(order, gt_idx_l, n_values)
    
    duration = (time.time() - local_start_time)
    print('(END) run time: %d:%02d'%(duration/60, duration%60))

  
    # log
    print("\nmAP: %.3f"%mAP)
    for i, n in enumerate(n_values):
        print("Recall@%d: %.3f"%(n, recalls[i]))
    duration = (time.time() - global_start_time)
    print('Global run time retrieval: %d:%02d'%(duration/60, duration%60))
    
    # write retrieval
    order_fn = "%s/%d_c%d_%d_order.txt"%(res_dir, args.slice_id, args.cam_id,
            args.survey_id)
    rank_fn = "%s/%d_c%d_%d_rank.txt"%(res_dir, args.slice_id, args.cam_id,
            args.survey_id)
    retrieval.write_retrieval(order, args.top_k, order_fn, rank_fn)

    # write perf
    perf_dir = 'res/%s/perf'%args.data
    np.savetxt(recalls_fn, np.array(recalls))
    np.savetxt(mAP_fn, np.array([mAP]))
    return mAP, recalls


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, required=True, help='Trial.')
    parser.add_argument('--dist_pos', type=float, required=True)
    parser.add_argument('--top_k', type=int, default=20)

    parser.add_argument("--lf_mode", type=str, 
            help="Local feature extraction method {sift, surf, orb, akaze, delf}")
    parser.add_argument('--max_num_feat', type=int)
    parser.add_argument("--agg_mode", type=str, 
            help="Aggregation mode {bow, vlad}")
    parser.add_argument('--vlad_norm', type=str)
    parser.add_argument('--n_words', type=int)
    parser.add_argument('--centroids', type=str)
    
    parser.add_argument('--data', type=str, required=True, help='{cmu_park, cmu_urban, lake}')
    parser.add_argument('--slice_id', type=int, default=22)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--survey_id', type=int, default=0)

    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--meta_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, default="")
    parser.add_argument('--des_dir', type=str, default="")
    
    parser.add_argument('--resize', type=int, default=0, help='set to 1 to resize img')
    parser.add_argument('--h', type=int, default=480, help='new height')
    parser.add_argument('--w', type=int, default=704, help='new width')
    args = parser.parse_args()

    n_values = [1, 5, 10, 20]
 
    res_dir = "res/%s/%d/retrieval/"%(args.agg_mode, args.trial)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    perf_dir = "res/%s/%d/perf/"%(args.agg_mode, args.trial)
    if not os.path.exists(perf_dir):
        os.makedirs(perf_dir)

    if args.data == "cmu_park" or args.data == "cmu_urban":
        args.survey_type = "cmu"
        kwargs = {"img_dir": args.img_dir, "seg_dir": ""}
    elif args.data == "symphony":
        args.survey_type = "symphony"
        kwargs = {"img_dir": args.img_dir, "seg_dir": "", 
                "mask_dir": args.mask_dir}
    else:
        raise ValueError("I don't know this dataset: %s"%args.data)

    centroids = np.loadtxt(args.centroids)
    bench(args, kwargs, centroids, n_values)

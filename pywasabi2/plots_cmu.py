"""Perf plots."""
import os

import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIG_SIZE = 6

plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
plt.rc('legend', fontsize=BIG_SIZE)

###############################################################################
# PAPER PLOTS

metric_l = ['mAP', 'rec1', 'rec5', 'rec10', 'rec20']
n_values = [1, 5, 10, 20]

method_l = [
        'random', 
        'wasabi', 'wasabi2',
        'netvlad', 'netvlad*', 
        'delf', 
        'vlad', 'vlad*',
        'bow', 'bow*',
        ]

color_l     = [
        'gray', # random
        'green', # wasabi
        'lime', # wasabi2
        'blue', 'red', # netvlad, netvlad*
        'orange',  # delf
        'hotpink', 'midnightblue', # vlad, vlad*
        'purple', 'crimson', # bow, bow*
        ]

fmt_l = ['.', 
        'o', 'o',
        '^', '^', 
        'x', 
        'v', 'v',
        '+', '+']

capsize_l = [1, 5, 5, 2, 3, '3', 4, 4]


def perf_rec_avg_surveys(perf, slice_v, cam_v, plot_dir, plot_fn):
    """Average the recalls over surveys for each slice. Shows the perf vs the
    semantic elements."""
    survey_num = perf['random']['mAP'].shape[0]
    slice_num = slice_v.shape[0]

    # avg over survey for each slice
    methods_num = len(method_l)
    for metric in metric_l[1:]:
        for method in method_l:
            #print('method: %s'%method)
            avg = np.zeros(slice_num)
            std = np.zeros(slice_num)
            for i, (slice_id, cam_id) in enumerate(zip(slice_v, cam_v)):
                avg[i] = np.mean(perf[method][metric][:,i])
                std[i] = np.std(perf[method][metric][:,i])
            
            perf[method]['avg_survey_%s'%metric] = avg
            perf[method]['std_survey_%s'%metric] = std

    # plot recall@N for each slice_cam
    for j, (slice_id, cam_id) in enumerate(zip(slice_v, cam_v)):
        plt.figure(figsize=(3,2))
        #plt.figure(figsize=(4,3))
        for i, method in enumerate(method_l):
            #print('method: %s'%method)
            color = color_l[i]
            
            avg_rec_l = []
            std_rec_l = []
            for metric in metric_l[1:]:
                avg_rec_l.append(perf[method]['avg_survey_%s'%metric][j])
                std_rec_l.append(perf[method]['std_survey_%s'%metric][j])
            avg_rec_v = np.array(avg_rec_l)
            std_rec_v = np.array(std_rec_l)
            plt.plot(n_values, avg_rec_v, label=method, color=color, marker=fmt_l[i])
            if method in ['wasabi', 'wasabi2']: # for better display
                alpha = 0.1
            else:
                alpha = 0.03
            plt.fill_between(n_values, avg_rec_v + std_rec_v, avg_rec_v -
                    std_rec_v, facecolor=color, alpha=alpha)
        
        fig_fn = '%s/recN_%d_c%d.png'%(plot_dir, slice_id, cam_id)
        plt.xlabel('N')
        plt.ylabel('Recall@N')
        plt.axis([0, 21, -.1, 1.1])
        plt.xticks(n_values)
        if j == 0:
            plt.legend(loc=0)
        plt.title('Slice %d - Camera %d'%(slice_id, cam_id))
        plt.savefig(fig_fn)
        plt.close()
    
    
    # assemble the plots (yes, I know, it could be more automatic)
    line_num = 4
    line_d = {}
    for line in range(line_num):
        line_d[line] = []
 
    for j, (slice_id, cam_id) in enumerate(
            [[22,0], [22,1], [23,0], [23,1], [24,0], [24,1], [25,0], [25,1]]):
        #print(slice_id, cam_id)
        fig_fn = "%s/recN_%d_c%d.png"%(plot_dir, slice_id, cam_id)
        fig = cv2.imread(fig_fn)
        if slice_id == 22: # shame on me for this manual assignment
            line = 0
        elif slice_id == 23:
            line = 1
        elif slice_id == 24:
            line = 2
        elif slice_id == 25:
            line = 3

        line_d[line].append(fig)

    # stack the plot lines
    img_line_d = {}
    for i in range(line_num):
        img_line_d[i] = np.hstack(line_d[i])
        #cv2.imshow("img_line_d[i]", img_line_d[i])
        #cv2.waitKey(0)
    # careful, it can stack them in non-sorted order
    out = np.vstack(list(img_line_d.values())) 
    cv2.imwrite(plot_fn, out)
    #cv2.imshow('out', out)
    #cv2.waitKey(0)
   

def perf_rec_avg_slices(perf, slice_v, cam_v, plot_dir, plot_fn):
    """Average the recalls over surveys for each slice. Shows the perf vs the
    semantic elements. """
    survey_num = perf['random']['mAP'].shape[0] # 9
    slice_num = slice_v.shape[0]
    
    # avg over survey for each slice
    methods_num = len(method_l)
    for metric in metric_l[1:]:
        for method in method_l:
            avg = np.zeros(survey_num)
            std = np.zeros(survey_num)
            for survey_id in range(survey_num):
                avg[survey_id] = np.mean(perf[method][metric][survey_id, :])
                std[survey_id] = np.std(perf[method][metric][survey_id, :])
            perf[method]['avg_survey_%s'%metric] = avg
            perf[method]['std_survey_%s'%metric] = std

    
    # plot recall@N for each slice_cam
    for survey_id in range(survey_num):
        plt.figure()
        for i, method in enumerate(method_l):
            color = color_l[i]
            avg_rec_l = []
            std_rec_l = []
            for metric in metric_l[1:]:
                avg_rec_l.append(perf[method]['avg_survey_%s'%metric][survey_id])
                std_rec_l.append(perf[method]['std_survey_%s'%metric][survey_id])
            avg_rec_v = np.array(avg_rec_l)
            std_rec_v = np.array(std_rec_l)
            plt.plot(n_values, avg_rec_v, label=method, color=color, marker=fmt_l[i])
            if method in ['wasabi', 'wasabi2']:
                alpha = 0.05
            else:
                alpha = 0.01
            plt.fill_between(n_values, avg_rec_v + std_rec_v, avg_rec_v -
                    std_rec_v, facecolor=color, alpha=alpha)
        
        fig_fn = "%s/recN_survey_%d.png"%(plot_dir, survey_id)
        plt.xlabel('N')
        plt.ylabel('Recall@N')
        plt.axis([0, 21, -.1, 1.1])
        plt.xticks(n_values)
        if survey_id == 0:
            plt.legend(loc=0)
        plt.title('Traversal %d'%survey_id)
        plt.savefig(fig_fn)
        plt.close()


    # assmble plots
    img_num_per_col = 3
    plot_num = survey_num
    line_num = int(np.ceil(plot_num/img_num_per_col))
    
    line_d = {}
    for line in range(line_num):
        line_d[line] = []
    
    for j in range(survey_num):
        fig_fn = "%s/recN_survey_%d.png"%(plot_dir, j)
        fig = cv2.imread(fig_fn)
        line = j//img_num_per_col
        #print(j, j//img_num_per_col)
        line_d[line].append(fig)

    # complete line2
    col_fill_num = line_num * img_num_per_col - plot_num
    fig = cv2.imread("%s/recN_survey_0.png"%(plot_dir))
    #print("line_num-1: %d"%(line_num-1))
    for _ in range(col_fill_num):
        line_d[line_num-1].append(255*np.ones(fig.shape, np.uint8))
    
    # stack the plot lines
    img_line_d = {}
    for i in range(line_num):
        img_line_d[i] = np.hstack(line_d[i])
        #cv2.imshow("img_line_d[i]", img_line_d[i])
        #cv2.waitKey(0)
    # careful, it can stack them in non-sorted order
    out = np.vstack(list(img_line_d.values())) 
    cv2.imwrite(plot_fn, out)
    #cv2.imshow('out', out)
    #cv2.waitKey(0)


def fig5():
    """Perf plot on cmu park."""
    slice_v = np.array([22, 23, 24, 25])
    cam_v = np.array([0, 1])
    slice_num = slice_v.shape[0]
    slice_v = np.tile(slice_v.reshape(slice_num,1), (1,2)).flatten()
    cam_v = np.tile(cam_v, (slice_num))
    
    perf = {}
    for i, method in enumerate(method_l):
        perf[method] = {}
        for metric in metric_l:
            if "*" in method:
                fn = "res/soa/cmu_park/%s_tuned/"%method[:-1]
            elif method == "wasabi2":
                fn = "res/cmu_park/"
            else:
                fn = "res/soa/cmu_park/%s/"%method
            #print(fn)
            perf[method][metric] = np.loadtxt("%s/%s.txt"%(fn, metric))
    
    #plot_dir = "res/tmp_plots/fig5/"
    #if not os.path.exists(plot_dir):
    #    os.makedirs(plot_dir)
    #perf_rec_avg_surveys(perf, slice_v, cam_v, plot_dir, "fig5_cmu_park.png")

    ### TODO: uncomment me to plot fig6
    #plot_dir = "res/tmp_plots/cvpr/img/fig6"
    plot_dir = "res/tmp_plots/fig6/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    perf_rec_avg_slices(perf, slice_v, cam_v, plot_dir, "fig6_cmu_park.png")



def fig9():
    """Perf plot on cmu urban."""
    slice_v = np.array([6, 7, 8])
    cam_v = np.array([0, 1])
    slice_num = slice_v.shape[0]
    slice_v = np.tile(slice_v.reshape(slice_num,1), (1,2)).flatten()
    cam_v = np.tile(cam_v, (slice_num))
    
    trial_l = [
            3, # random
            19, # wasabi
            11, -1, # netvlad
            35, # delf
            19, -1,# vlad
            26, -1, # bow
            ]

    perf = {}
    survey_num = 10 # TODO: delete once you delete survey 10
    for i, method in enumerate(method_l):
        perf[method] = {}
        perf[method]['trial'] = trial_l[i]
        if trial_l[i] == -1:
            continue
        perf[method]['res_dir'] = res_dir_l[i]
        for metric in metric_l:
            perf[method][metric] = np.loadtxt('%s/%d/%s.txt'%(res_dir_l[i],
                trial_l[i], metric))[:survey_num, :]

    #plot_dir = "plots/img/fig9"
    #if not os.path.exists(plot_dir):
    #    os.makedirs(plot_dir)
    #perf_rec_avg_surveys(perf, slice_v, cam_v, plot_dir, "%s.png"%plot_dir)

    # here, all surveys have ground-truth camera poses
    survey_ok = np.ones(slice_v.shape[0], np.int32)
    plot_dir = "plots/img/fig10"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    perf_rec_avg_slices(perf, survey_ok, slice_v, cam_v, plot_dir, "%s.png"%plot_dir)


if __name__=='__main__':
    fig5() # park
    #fig9() # urban

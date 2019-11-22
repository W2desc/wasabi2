import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# PAPER PLOTS


metric_l = ['mAP', 'rec1', 'rec5', 'rec10', 'rec20']
n_values = [1, 5, 10, 20]
survey_num = 10

method_l    = ['random', 
        'wasabi', 'wasabi2',
        'netvlad', 'netvlad*', 
        'delf', 
        'vlad', 'vlad*',
        'bow', 'bow*']

color_l     = [
        'gray', # random
        'green', # wasabi
        'lime', # wasabi2
        'blue', 'red', # netvlad, netvlad*
        'orange',  # delf
        'hotpink', 'midnightblue', # vlad, vlad*
        'purple', 'crimson', # bow, bow*
        ]


trial_l     = [1, 
        15, 50,
        10, 12, 
        36, 30, 
        33, 
        29, 32]
#res_dir_l   = ['res/random/', 
#        'res/wasabi/', 'res/vlad',
#        'res/netvlad/', 'res/netvlad/',
#        'res/vlad/', # delf
#        'res/vlad/', # vlad
#        'res/vlad/', # vlad*
#        'res/vlad/', # bow
#        'res/vlad/'] # bow*'


fmt_l = ['.', 
        'o', 'o',
        '^', '^', 
        'x', 
        'v', 'v',
        '+', '+']
capsize_l = [1, 5, 2, 3, '3', 4, 4]

perf = {}
for i, method in enumerate(method_l):
    perf[method] = {}
    for metric in metric_l:
        if "*" in method:
            fn = "res/soa/symphony/%s_tuned/"%method[:-1]
        elif method == "wasabi2":
            fn = "res/symphony/"
        else:
            fn = "res/soa/symphony/%s/"%method
        perf[method][metric] = np.loadtxt("%s/%s.txt"%(fn, metric))


###############################################################################
def perf_rec_global():
    """Global recall@N over all images."""
    # avg over survey for each slice
    methods_num = len(method_l)
    for metric in metric_l[1:]:
        for method in method_l:
            perf[method]['avg_%s'%metric] = np.mean(perf[method][metric])
            perf[method]['std_%s'%metric] = np.std(perf[method][metric])
    
    # plot recall@N for each slice_cam
    plt.figure()
    for i, method in enumerate(method_l):
        color = color_l[i]
        avg_rec_l = []
        std_rec_l = []
        for metric in metric_l[1:]:
            avg_rec_l.append(perf[method]['avg_%s'%metric])
            std_rec_l.append(perf[method]['std_%s'%metric])
        avg_rec_v = np.array(avg_rec_l)
        std_rec_v = np.array(std_rec_l)
        plt.plot(n_values, avg_rec_v, label=method, color=color,
                marker=fmt_l[i])
        if method in ['wasabi', 'netvlad', 'netvlad*']:
            alpha = 0.1
        else:
            alpha = 0.03
        plt.fill_between(n_values, avg_rec_v + std_rec_v, avg_rec_v -
                std_rec_v, facecolor=color, alpha=alpha)
    
    fig_fn = 'fig7_symphony.png'
    plt.xlabel('N')
    plt.ylabel('Recall@N')
    plt.axis([0, 21, -.1, 1.1])
    plt.xticks(n_values)
    plt.legend(loc=4)
    plt.title('Global symphony recall@N')
    plt.savefig(fig_fn)
    plt.close()
    
    out = cv2.imread(fig_fn)
    cv2.imshow('out', out)
    cv2.waitKey(0)


def perf_mAP_global():
    """Global recall@N over all images."""
    # avg over survey for each slice
    methods_num = len(method_l)
    metric='mAP'
    for method in method_l:
        print('method: %s'%method)
        perf[method]['avg_%s'%metric] = np.mean(perf[method][metric])
        perf[method]['std_%s'%metric] = np.std(perf[method][metric])

    
    # plot recall@N for each slice_cam
    plt.figure()
    mean_v = []
    std_v = []
    for i, method in enumerate(method_l):
        mean_v.append(perf[method]['avg_mAP'])
        std_v.append(perf[method]['std_mAP'])
    
    absc = np.arange(len(method_l))
    #plt.errorbar(absc, mean_v, std_v, linestyle='None', fmt='o',
    #        color=color_l[i], alpha=1, capsize=2)
 
    for i, method in enumerate(method_l):
        plt.errorbar(i+1, mean_v[i], std_v[i], linestyle='None', fmt=fmt_l[i],
                color=color_l[i], alpha=1, label=method,
                capsize=capsize_l[i])
    
    fig_fn = 'plots_cvpr/img/symphony_mAP_global.png'
    plt.xlabel('Method')
    plt.ylabel('Global mAP over symphony')
    #plt.axis([0, 21, -.1, 1.1])
    #plt.xticks(method_l)
    plt.xticks([])
    plt.legend(loc=0)
    plt.title('Global symphony mAP')
    plt.savefig(fig_fn)
    plt.close()
    
    out = cv2.imread(fig_fn)
    cv2.imshow('out', out)
    cv2.waitKey(0)



if __name__=='__main__':
    
    perf_rec_global()
    #perf_mAP_global()

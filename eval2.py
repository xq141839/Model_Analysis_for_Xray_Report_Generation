from nlgeval import compute_metrics
metrics_dict = compute_metrics(hypothesis='outputs/NLMCXR_ClsGenInt_DenseNet121_MaxView2_NumLabel114_NoHistory_Hyp.txt',
                               references=['outputs/NLMCXR_ClsGenInt_DenseNet121_MaxView2_NumLabel114_NoHistory_Ref.txt'])
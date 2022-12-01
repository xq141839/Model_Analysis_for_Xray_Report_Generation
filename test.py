from nlgeval import compute_metrics
from torchmetrics.text.rouge import ROUGEScore
import torchmetrics.functional as F
from pprint import pprint

hyp_path = 'small/tranformer_mv_Hyp.txt'
ref_path = 'small/tranformer_mv_Ref.txt'

with open(hyp_path) as file:
                hyp_ls = [sentence.strip() for sentence in file.readlines()]
                hyp_str = ' '.join(hyp_ls)
                hyp_sp = hyp_str.split('.')

with open(ref_path) as file:
                ref_ls = [sentence.strip() for sentence in file.readlines()]
                ref_str = ' '.join(ref_ls)
                ref_sp = ref_str.split('.')


# print(hyp_sp)
# print('------------------------------------')
# print(ref_sp)

rouge = ROUGEScore()
pprint(rouge(hyp_sp, ref_sp))
print(F.word_error_rate(preds=hyp_sp, target=ref_sp))
print(F.match_error_rate(preds=hyp_sp, target=ref_sp))

metrics_dict = compute_metrics(hypothesis=hyp_path,
                               references=[ref_path])

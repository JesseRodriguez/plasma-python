from plasma.utils.datasets import DatasetBuilder
from plasma.conf import conf

sig_permute = ['q95', 'iptarget', 'ip', 'energy', 'lm', 'etemp_profile']#,\
#        'edens_profile', 'ecei_hfs']
sig_fixed = ['ecei']

DB = DatasetBuilder(conf)
DB.Check_Signal_Group_Permute(sig_permute, sig_fixed, verbose = False, cpu_use = 0.6)
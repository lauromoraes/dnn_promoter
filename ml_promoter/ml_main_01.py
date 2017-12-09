# -*- coding: utf-8 -*-

def main0():
    from ml_experiment import BaseExperiment
    from ml_model import Model_Hist_Conv1
    
    print('> Running MAIN 0\n')
    
    npath = "fasta/Bacillus_non_prom.fa"
    ppath = "fasta/Bacillus_prom.fa"
    
#    archtecture = Arch_01()
#    archtecture.setup_data(npath, ppath)
    
    model = Model_Hist_Conv1(npath, ppath)
    
    exp = BaseExperiment(model)
    
    exp.error_metric = exp.errors[10]
    exp.batch_size = 64
    exp.epochs = 300
    exp.optimizer = 'adam'    
    exp.validation_split = 0.1
    exp.class_weigths = {
            0: 0.25,
            1: 0.75
    }
    
    exp.run_grid()
#    exp.run()

def main02():
    from ml_architecture import Arch_Feedforward_001
    
    npath = "fasta/Bacillus_non_prom.fa"
    ppath = "fasta/Bacillus_prom.fa"
    
    arch = Arch_Feedforward_001()
    arch.setup_data(npath, ppath)
    arch.run_grid()
    
    

if __name__ == "__main__":
    
    print(''.join(('='*20, '<START>', '='*20)))
    TEST = 0
    if TEST==0:
        main02()
    else:
        print('>>> ERROR: invalid test module ({})'.format(TEST))
    print(''.join(('='*20, '<END>', '='*20)))
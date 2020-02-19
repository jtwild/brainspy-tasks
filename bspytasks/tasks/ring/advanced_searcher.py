import torch
import matplotlib.pyplot as plt

from bspyproc.bspyproc import get_processor
from bspytasks.tasks.ring.classifier import RingClassificationTask as Task
from bspyproc.utils.pytorch import TorchUtils


class AdvancedRingSearcher():
    def __init__(self, configs):
        self.configs = configs
        self.processor = get_processor(configs['algorithm_configs']['processor'])
        self.task = Task(configs)

    def improve_solution(self, results, model):
        self.processor.load_state_dict(model.copy())
        inputs = TorchUtils.get_tensor_from_numpy(results['inputs'])
        targets = TorchUtils.get_tensor_from_numpy(results['targets'])
        TorchUtils.init_seed(results['seed'], deterministic=True)
        new_results = self.task.run_task(inputs, targets, results['mask'])

        plt.figure()
        plt.plot(results['best_output'])
        plt.plot(new_results['best_output'])
        plt.show()


if __name__ == '__main__':
    import pickle
    import torch
    import matplotlib.pyplot as plt
    from bspyalgo.utils.io import load_configs

    model = torch.load('model.pth')
    results = pickle.load(open('best_output_results.pkl', "rb"))
    configs = load_configs('ring_classification_configs.json')
    searcher = AdvancedRingSearcher(configs)
    searcher.improve_solution(results, model)

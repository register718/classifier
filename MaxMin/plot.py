import torch
import trainer
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import signal

class Plot:

    def __init__(self, trainer: trainer.Trainer, num_targets: int, xlim=1, ylim=1, max_per_target: int=3):
        self.trainer = trainer
        self.fig, self.ax = plt.subplots()
        self.max_per_target = max_per_target
        self.scatters = [self.ax.scatter(0, 0, s=5, label=f"Data {str(i)}", c=self.__genColor__()) for i in range(num_targets)]
        self.ax.set(xlim=[-1, 1], ylim=[-1, 1])
        self.ax.legend()
        self.num_targets = num_targets

    def __genColor__(self):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
        return color
    
    def update(self, frame, stop=True):
        #training
        print("Epoch", frame)
        self.trainer.train_Epoch()
        #plotting
        points = self.trainer.test_Epoch() # [TARGET, DATA]
        scatterLists = [[] for i in range(self.num_targets)]
        scatterListsFull = [False for i in range(self.num_targets)]
        countFullScatters = 0
        idx = -1
        totalLoss = 0
        while countFullScatters < self.num_targets and idx < len(points) - 1:
            idx += 1
            target, loss, y = points[idx]
            totalLoss += loss
            if len(scatterLists[target]) >= self.max_per_target:
                if not scatterListsFull[target]:
                    print(y)
                    scatterListsFull[target] = True
                    countFullScatters += 1
                continue
            for ba in y:
                scatterLists[target].append(ba.unsqueeze(dim=0))
        print(totalLoss)
        
        ## PLOT ##
        for idx, sc in enumerate(scatterLists):
            arr = torch.cat(sc, dim=0).numpy()
            if len(arr.shape) == 2 and arr.shape[1] == 2:
                self.scatters[idx].set_offsets(arr)
        ## STOP ##
        if stop and frame >= self.trainer.epochs - 1:
            self.ani.event_source.stop()
            plt.close()
        return self.scatters



    def start_animation(self):
        print("Training of ", self.trainer.epochs, "epochs")
        self.ani = animation.FuncAnimation(fig=self.fig, func=self.update, frames=self.trainer.epochs, interval=0)
        plt.show(block=True)
    
    def start_without_animation(self):
        self.stop = False
        def sigHandler(sig, frame):
            self.stop = True
        signal.signal(signal.SIGINT, sigHandler)
        for i in range(self.trainer.epochs):
            if self.stop:
                break
            self.update(i, stop=False)
        return

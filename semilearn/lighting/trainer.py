# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
from progress.bar import Bar

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from semilearn.core.utils import get_optimizer, get_cosine_schedule_with_warmup, get_logger, EMA



class Trainer:
    def __init__(self, config, algorithm, verbose=0):
        self.config = config
        self.verbose = verbose
        self.algorithm = algorithm

        # TODO: support distributed training?
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            self.algorithm.model = self.algorithm.model.cuda(config.gpu)

        # setup logger
        self.save_path = os.path.join(config.save_dir, config.save_name)
        self.logger = get_logger(config.save_name, save_path=self.save_path, level="INFO")


    def fit(self, train_lb_loader, train_ulb_loader, eval_loader):
        self.algorithm.loader_dict = {
            'train_lb': train_lb_loader,
            'train_ulb': train_ulb_loader,
            'eval': eval_loader
        }
        self.algorithm.model.train()
        # train
        self.algorithm.it = 0
        self.algorithm.best_eval_acc = 0.0
        self.algorithm.best_epoch = 0
        self.algorithm.call_hook("before_run")

        for epoch in range(self.config.epoch):
            self.algorithm.epoch = epoch
            print("Epoch: {}".format(epoch))
            if self.algorithm.it > self.config.num_train_iter:
                break

            bar = Bar('Processing', max=len(train_lb_loader))

            self.algorithm.model.train()
            self.algorithm.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(train_lb_loader, train_ulb_loader):

                if self.algorithm.it > self.config.num_train_iter:
                    break
                
                self.algorithm.call_hook("before_train_step")
                out_dict, log_dict = self.algorithm.train_step(**self.algorithm.process_batch(**data_lb, **data_ulb))
                self.algorithm.out_dict = out_dict
                self.algorithm.log_dict = log_dict
                self.algorithm.call_hook("after_train_step")
                
                bar.suffix = ("Iter: {batch:4}/{iter:4}.".format(batch=self.algorithm.it, iter=len(train_lb_loader)))
                bar.next()
                self.algorithm.it += 1
            bar.finish()

            self.algorithm.call_hook("after_train_epoch")
            
            # validate
            result = self.evaluate(eval_loader)

            # save model
            self.algorithm.save_model('latest_model.pth', self.save_path)

            # best
            if result['acc'] > self.algorithm.best_eval_acc:
                self.algorithm.best_eval_acc = result['acc']
                self.algorithm.best_epoch = self.algorithm.epoch
                self.algorithm.save_model('model_best.pth', self.save_path)
        
        self.logger.info("Best acc {:.4f} at epoch {:d}".format(self.algorithm.best_eval_acc, self.algorithm.best_epoch))
        self.logger.info("Training finished.")


    def evaluate(self, data_loader, use_ema_model=False):
        y_pred, y_logits, y_true = self.predict(data_loader, use_ema_model, return_gt=True)
        top1 = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.logger.info("confusion matrix")
        self.logger.info(cf_mat)
        result_dict = {'acc': top1, 'precision': precision, 'recall': recall, 'f1': f1}
        self.logger.info("evaluation metric")
        for key, item in result_dict.items():
            self.logger.info("{:s}: {:.4f}".format(key, item))
        return result_dict


    def predict(self, data_loader, use_ema_model=False, return_gt=False):
        self.algorithm.model.eval()
        if use_ema_model:
            self.algorithm.ema.apply_shadow()

        y_true = []
        y_pred = []
        y_logits = []
        with torch.no_grad():
            for data in data_loader:
                logits = self.algorithm.get_logits(data, 'logits')
                y = self.algorithm.get_targets(data)

                y_true_list = y.cpu().tolist()
                y_pred_list = torch.max(logits, dim=-1)[1].cpu().tolist()
                y_logits_list = torch.softmax(logits, dim=-1).cpu().tolist()
                    
                y_true.extend(y_true_list)
                y_pred.extend(y_pred_list)
                y_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())

                for i in range(len(y_true_list)):
                    sample_id = data['sample_id'][i]
                    self.logger.info(f"Sample[{sample_id}] --> Predicted class={y_true_list[i]}  Actual class={y_pred_list[i]}  Logits={y_logits_list[i]}")

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)

        if use_ema_model:
            self.algorithm.ema.restore()
        self.algorithm.model.train()
        
        if return_gt:
            return y_pred, y_logits, y_true
        else:
            return y_pred, y_logits
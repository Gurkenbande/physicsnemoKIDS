import torch
import torch.nn as nn
import numpy as np
from metrics import AverageMeter, accuracy_binary_one, accuracy_binary_one_classes
from tools import write_log

#-----------------------------------------------------
#---------------------- TRAIN ------------------------
#-----------------------------------------------------


class Trainer(object):

    def __init__(self):
        super(Trainer, self).__init__()

    def _create_plots_C(self, y_pred, y, t, train_mask, graph, accelerator, step, args):
        pass

    def _create_plots_R_Rall(self, y_pred, y, t, train_mask, graph, accelerator, step, args):
        pass

                
    #--- REGRESSOR (either R or Rall)

    def train_R_Rall(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=0, log_freq=5):
        
        write_log(f"\nStart training the regressor.", args, accelerator, 'a')
        step = 0
      #  MSELoss = nn.MSELoss()

        for epoch in range(epoch_start, epoch_start + args.epochs):

            print(f"\n===== Epoch {epoch+1} =====")
            model.train()

            loss_meter = AverageMeter()
            val_loss_meter = AverageMeter()
         #   val_loss_term1_meter = AverageMeter()
          #  val_loss_term2_meter = AverageMeter()
            print("Length of DataLoader: ",len(dataloader_train))

            # -------------------------------------
            # TRAIN LOOP
            # -------------------------------------
            for batch_i, graph in enumerate(dataloader_train):
                
                optimizer.zero_grad(set_to_none=True)

                y_pred = model(graph)

                train_mask = graph['high'].train_mask
                y = graph['high'].y
                y_pred = y_pred[train_mask]
                y = y[train_mask]
                loss = loss_fn(y_pred, y)

                accelerator.backward(loss)

                optimizer.step()

                loss_meter.update(float(loss.detach()), n=y_pred.shape[0])
                if batch_i % 500 == 0:
                    print(f"[Epoch {epoch} Batch {batch_i}] Loss: {loss.item():.4f}")

                if batch_i == len(dataloader_train) - 1:
                    print(f"[Epoch {epoch}] Avg Train Loss: {loss_meter.avg:.4f}")

           #     accelerator.log({
            #        'epoch': epoch,
             #       'train loss iteration': loss_meter.val,
              #      'train loss avg': loss_meter.avg
               # }, step=step)

                step += 1

            # -------------------------------------
            # VALIDATION
            # -------------------------------------
            print("\n===== Starte Validation =====")
            model.eval()

            if epoch % log_freq == 0:
                y_pred_list = []
                y_list = []
                train_mask_list = []
                t_list = []

            with torch.no_grad():
                print("Length of DataLoader: ",len(dataloader_val))
                for batch_i, graph in enumerate(dataloader_val):
                   
                    y_pred = model(graph)
                    train_mask = graph['high'].train_mask
                    y = graph['high'].y
                  #  w = graph['high'].w

                    yp_m = y_pred[train_mask]
                  #  y_m = y[train_mask]
                   # w_m = w[train_mask]

                  #  loss_mse = MSELoss(yp_m, y_m)
                  #  loss_qmse = loss_fn(yp_m, y_m, w_m)
                  #  loss = loss_mse + args.alpha * loss_qmse
                    loss = loss_fn(y_pred, y)

                    val_loss_meter.update(float(loss.detach()), n=yp_m.shape[0])
                  #  val_loss_term1_meter.update(float(loss_mse.detach()), n=yp_m.shape[0])
                  #  val_loss_term2_meter.update(float(loss_qmse.detach()), n=yp_m.shape[0])
 
                    if batch_i % 500 == 0:
                        print(f"[Epoch {epoch} Batch {batch_i}] Loss: {loss.item():.4f}")

                    if batch_i == len(dataloader_val) - 1:
                        print(f"[Epoch {epoch}] Avg Val Loss: {loss_meter.avg:.4f}")

             #       accelerator.log({
              #          'epoch': epoch,
               #         'val loss iteration': val_loss_meter.val,
                #        'val loss avg': val_loss_meter.avg
                 #   }, step=step)

                    if epoch % 5 == 0:
                        t = graph.t
                        y_pred_g, y_g, mask_g, t_g = accelerator.gather(
                            (y_pred.unsqueeze(0), y.unsqueeze(0), train_mask.unsqueeze(0), t)
                        )
                        y_pred_list.append(torch.atleast_2d(y_pred_g))
                        y_list.append(torch.atleast_2d(y_g))
                        train_mask_list.append(torch.atleast_2d(mask_g))
                        t_list.append(torch.atleast_2d(t_g))

                if epoch % 5 == 0:
                    t = torch.cat(t_list, dim=1).squeeze()
                    y_pred = torch.cat(y_pred_list, dim=0).swapaxes(0, 1)
                    y = torch.cat(y_list, dim=0).swapaxes(0, 1)
                    train_mask = torch.cat(train_mask_list, dim=0).swapaxes(0, 1)
                    self._create_plots_R_Rall(y_pred, y, t, train_mask, graph, accelerator, step, args)


            if "quantized_loss" in args.loss_fn:
                accelerator.log({
                    'epoch': epoch,
                    'val loss avg': val_loss_meter.avg,
                  #  'val mse loss avg': val_loss_term1_meter.avg,
                  #  'val qmse loss avg': val_loss_term2_meter.avg
                }, step=step)
            else:
                accelerator.log({'epoch': epoch, 'val loss avg': val_loss_meter.avg}, step=step)

            if lr_scheduler is not None:
                lr_scheduler.step()


#-----------------------------------------------------
#----------------------- TEST ------------------------
#-----------------------------------------------------


class Tester(object):

    def test(self, model, dataloader, args, accelerator=None):
        model.eval()
        step = 0 

        pr = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred = model(graph)
                if args.model_type == "R" or args.model_type == "Rall":
                    y_pred = torch.where(torch.isfinite(torch.expm1(y_pred)), torch.expm1(y_pred), np.nan)
                elif args.model_type == "C":
                    y_pred = torch.where(y_pred < 0, 1, 0)
                pr.append(y_pred)
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr = torch.stack(pr)
        times = torch.stack(times)

        return pr, times

    '''def test_RC(self, model_R, model_C, dataloader, args, accelerator=None):
        model_R.eval()
        model_C.eval()
        step = 0 

        pr_R = []
        pr_C = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred_R = model_R(graph)
                y_pred_R = torch.where(torch.isfinite(torch.expm1(y_pred_R)), torch.expm1(y_pred_R), np.nan)
                pr_R.append(y_pred_R)
                
                y_pred_C = model_C(graph)
                y_pred_C = torch.where(y_pred_C < 0, 1, 0)
                pr_C.append(y_pred_C)
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr_R = torch.stack(pr_R)
        pr_C = torch.stack(pr_C)
        times = torch.stack(times)

        return pr_R, pr_C, times'''
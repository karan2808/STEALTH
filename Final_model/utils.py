import torch
import torch.nn as nn
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_valid(train_dataloader, val_dataloader, num_epochs, myModel, mySup_Res, optimizer, scheduler, save_path, super_res_path):

  v_loss = 10000

  # loss function, l1 loss with smoothness penalty
  loss_function = torch.nn.SmoothL1Loss(size_average=True)

  # Optimzer, loss function and scheduler for SR module

  optimizer1     = torch.optim.Adam(mySup_Res.parameters(), lr = learning_rate)
  loss_function1 = torch.nn.MSELoss(size_average = None, reduce = None, reduction = 'mean')
  scheduler1     = torch.optim.lr_scheduler.StepLR(optimizer1, gamma=0.8, step_size=300, verbose = True)

  for epoch in range(num_epochs):

    running_loss, num_sequence = 0, 0
    train_loss = 0
    myModel.train(True)
    mySup_Res.train(True)

    for i, (left_img, right_img, disp_img, disp_ups) in enumerate(train_dataloader):

      optimizer.zero_grad()
      optimizer1.zero_grad()

      # the images go on the gpu
      left_img       = left_img.to(device)
      right_img      = right_img.to(device)
      disp_img_gt    = disp_img.to(device)
      disp_ups       = disp_ups.to(device)

      disp_img_pred  = myModel.forward(left_img, right_img)
      loss = loss_function(disp_img_pred, disp_img_gt)
      loss.backward()
      optimizer.step()

      del left_img
      del right_img

      if epoch < 250:
        disp_img_pred_res = mySup_Res(disp_ups)
      else:
        disp_img_pred_res = mySup_Res(disp_img_pred.detach())
      loss1             = loss_function1(disp_img_pred_res, disp_img_gt)
      loss1.backward()
      optimizer1.step()

      running_loss = running_loss + loss.item()
      running_loss_supres = running_loss_supres + loss1.item()
      torch.cuda.empty_cache()

    print('EPOCH = ' + str(epoch))
    
    print('Training Loss = ' + str(running_loss/len(train_set)))

    print('Training Loss (Super Resolution = ' + str(running_loss_supres/len(train_set)))

    # Perform validation on validation set
    with torch.no_grad():
      val_loss = validate(val_dataloader, myModel, loss_function)

    scheduler.step()
    scheduler1.step()
    
    # Save model with least validation loss
    if val_loss < v_loss:
      v_loss = val_loss
      torch.save(myModel.state_dict(),save_path)
      torch.save(mySup_Res.state_dict(),super_res_path)

    torch.cuda.empty_cache()

def validate(val_dataloader, model, loss_function):
  model.eval()
  running_loss, num_sequence = 0, 0

  with torch.no_grad():
    for i, (left_img, right_img, disp_img) in enumerate(val_dataloader):

      # the images go on the gpu
      left_img       = left_img.to(device)
      right_img      = right_img.to(device)
      disp_img_gt    = disp_img.to(device)

      disp_img_pred  = model.forward(left_img, right_img)
      loss           = loss_function(disp_img_pred, disp_img_gt)

      num_sequence, running_loss = num_sequence + 1, running_loss + loss.item()

      # Optimization snippet:
      del left_img,right_img

  print('Validation Loss = ' + str(running_loss/len(valid_set)))
  return (running_loss/len(valid_set))


# Saves predictions at location 'eval_path':
def evaluate(testdata_loader, model, super_res_model, test_names, eval_path):
  model.eval()
  super_res_model.eval()

  with torch.no_grad():
    for i, (left_img, right_img, disp_img) in enumerate(test_dataloader):

      f = open(test_names, 'r')
      m = 0

      # the images go on the gpu
      left_img = left_img.to(device)
      right_img = right_img.to(device)

      disp_img_pred = model(left_img, right_img)
      disp_img_pred = super_res_model(disp_img_pred)

      for j in range(disp_img_pred.shape[0]):
        disp_pred = disp_img_pred[j,:,:].detach().cpu().numpy()
        disp_pred[disp_pred < 0] = 0          # Perform clipping operation on predictions.
        disp_pred[disp_pred > 192.0] = 192.0
        plt.imsave(eval_path + f[m].split('\n')[0] ,disp_pred,cmap='gray')
        m += 1


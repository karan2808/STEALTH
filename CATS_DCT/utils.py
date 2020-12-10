import torch
import torch.nn as nn
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_valid(train_dataloader, val_dataloader, num_epochs, myModel, optimizer, scheduler, save_path):

  v_loss = 10000

  # loss function, l1 loss with smoothness penalty
  loss_function = torch.nn.SmoothL1Loss(size_average=True)

  for epoch in range(num_epochs):

    running_loss, num_sequence = 0, 0
    train_loss = 0
    myModel.train(True)

    for i, (left_img, right_img, disp_img) in enumerate(train_dataloader):

      optimizer.zero_grad()

      # the images go on the gpu
      left_img       = left_img.to(device)
      right_img      = right_img.to(device)
      disp_img_gt    = disp_img.to(device)

      disp_img_pred  = myModel.forward(left_img, right_img)
      loss = loss_function(disp_img_pred, disp_img_gt)
      loss.backward()
      optimizer.step()

      running_loss = running_loss + loss.item()

      torch.cuda.empty_cache()

    print('EPOCH = ' + str(epoch))
    
    print('Training Loss = ' + str(running_loss/len(train_set)))

    # Perform validation on validation set
    with torch.no_grad():
      val_loss = validate(val_dataloader, myModel, loss_function)

    if (num_epochs == 500)
      scheduler.step()
    
    # Save model with least validation loss
    if val_loss < v_loss:
      v_loss = val_loss
      torch.save(myModel.state_dict(),save_path)

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
def evaluate(testdata_loader, model, test_names, eval_path):
  model.eval()

  with torch.no_grad():
    for i, (left_img, right_img, disp_img) in enumerate(test_dataloader):

      f = open(test_names, 'r')
      m = 0

      # the images go on the gpu
      left_img = left_img.to(device)
      right_img = right_img.to(device)

      disp_img_pred = model(left_img, right_img)

      for j in range(disp_img_pred.shape[0]):
        disp_pred = disp_img_pred[j,:,:].detach().cpu().numpy()
        disp_pred = cv2.idct(disp_pred)
        disp_pred[disp_pred < 0] = 0          # Perform clipping operation on predictions.
        disp_pred[disp_pred > 192.0] = 192.0
        plt.imsave(eval_path + f[m].split('\n')[0] ,disp_pred,cmap='gray')
        m += 1


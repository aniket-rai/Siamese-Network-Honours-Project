import torch
import gc


def train(model, optimiser, train_loader, val_loader, epochs, criterion, scheduler):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  train_losses = []
  val_losses = []
  cur_step = 0

  for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    print("Starting epoch " + str(epoch+1))
    x = 1;

    for img1, img2, labels in train_loader:
      # Forward
      img1 = img1.to(device)
      img2 = img2.to(device)
      labels = labels.to(device)
      out1, out2 = model(img1, img2)
      loss = criterion(out1, out2, labels)
      
      # Backward and optimize
      optimiser.zero_grad()
      loss.backward()
      optimiser.step()

      # Delete intermediate variables
      del img1, img2, labels, out1, out2

      # Calculate total loss
      running_loss += loss.item()
      torch.cuda.empty_cache()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    val_running_loss = 0.0
    
    # check validation loss after every epoch
    with torch.no_grad():
      model.eval()
      for img1, img2, labels in val_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, labels)
        val_running_loss += loss.item()

    avg_val_loss = val_running_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)
    # if (epoch % 5 == 0):
    #   torch.save(model.state_dict(), f"/content/drive/Shareddrives/Project 3- Face Recognition -2021/Aniket Google Colab Notebook/siamese_network_epoch_{epoch}.pth")
    
    print(f"Epoch [{epoch+1}/{epochs}],Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.8f}")

  print("Finished Training")  
  
  print("Saving Model...")
  torch.save(model.state_dict(), '/content/drive/Shareddrives/Project 3- Face Recognition -2021/Aniket Google Colab Notebook/siamese_network3.pth')
  print("Model Saved!")
  
  return train_losses, val_losses
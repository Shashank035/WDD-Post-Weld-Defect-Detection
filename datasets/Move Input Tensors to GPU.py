def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for epoch in range(50):
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to the GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')

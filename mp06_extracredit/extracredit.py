import torch, random, math, json
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE=torch.float32
DEVICE=torch.device("cpu")

###########################################################################################
def trainmodel():
    # Well, you might want to create a model a little better than this...
    #model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(in_features=8*8*15, out_features=1))
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features = 8*8*15, out_features = 450),
        torch.nn.ReLU(),
        torch.nn.Linear(450, 164),
        torch.nn.ReLU(),
        torch.nn.Linear(164, 132),
        torch.nn.ReLU(),
        torch.nn.Linear(132, 1)
    )

    # ... and if you do, this initialization might not be relevant any more ...
    #model[1].weight.data = initialize_weights()
    #model[1].bias.data = torch.zeros(1)
    #loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    for epoch in range(2000):
        for x,y in trainloader:
            #pass # Replace this line with some code that actually does the training
            labels_pred = model(x)
            loss = loss_fn(labels_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ... after which, you should save it as "model.pkl":
    torch.save(model, 'model.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()
    

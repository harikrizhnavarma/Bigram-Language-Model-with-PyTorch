import torch
import torch.nn as nn
import torch.optim as optim
import bigram_dataprep
from torch.nn import functional as F


# instantiate the data preparation script to obj
obj = bigram_dataprep.DataPrep(dataset_name = 'Bigram_Language_Model\input.txt') 

# write the neural netork
class BigramNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # create an embedding table of dimension vicabulary size (65 in our case)
        self.bigram_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, input):

        #calculating the logit values to predict next character by inputting a random character
        logits = self.bigram_embedding_table(input)
        return logits

# instantiating the model
model = BigramNN(vocab_size = obj.vocab_size)

# create an adam optimizer
optimizer = optim.AdamW(model.parameters(), lr = 0.001)

#training loop

epochs = 20000
least_loss = None
best_model = None
for epoch in range(epochs):

    #retrieve input_data, and its targets from the dataset
    input_data, target_data = obj.get_batch(split = 'train', block_size = 8, batch_size = 32)

    # reset the gradients to zero
    optimizer.zero_grad(set_to_none = True)

    #forward passing
    logits = model.forward(input = input_data)
    B,T,C = logits.shape 
    logits = logits.view(B*T, C) # reshape the logits tensor to 2D
    target_data = target_data.view(B*T) # match the shape of logits tensor

    #calculating loss and back propogating
    loss = F.cross_entropy(logits, target_data)
    
    # saving the best model and its loss
    if least_loss is None:
        least_loss = loss.item()
    elif loss < least_loss:
        least_loss = loss.item()
        best_model = model.state_dict() # saving th model with least loss value
    
    loss.backward() # back propogating
    optimizer.step()

    #print(loss.item()) if epoch == 0 or epoch%2000 == 0 else None
print(f'Training Finished. Least Loss value:{least_loss}')

#load the best model
model.load_state_dict(best_model)

# defining function to generate next most probable token using the model we trained
def generate(logit, input, max_new_tokens):
    for i in range(max_new_tokens):
        logit = model.forward(input) # input the new data
        probs = F.softmax(logit[:, -1, :], dim = -1) # indexing only to get the logits value of last token in the sequence since bigram only looks at the last character.
                                                     #Calculate softmax probability.
        next_token = torch.multinomial(probs, num_samples = 1) #retrieve the token with highest probability
        input = torch.cat((input, next_token), dim = 1) # concat new token with previous token
    
    return input

# generating encoded text
output = generate(logit = logits, input = torch.tensor([[47, 44, 50, 50, 53]]), max_new_tokens = 500) # input is encoded tensor of word 'hello'.

# decoding the tokens 
decoded_output = obj.decode(output[0].tolist())

print(decoded_output)
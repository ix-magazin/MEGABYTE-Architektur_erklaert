# Listing 1: Laderoutine für die englische Wikipedia im MEGABYTE-Nachbau
with gzip.open('./data/enwik8.gz') as file:
  x = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
  train_x, valid_x = np.split(x, [int(90e6)])
  data_train, data_val = map(torch.from_numpy, (train_x, valid_x))

# Listing x:
from datasets import load_dataset

# Load the German Wikipedia dataset
dataset = load_dataset("wikipedia", "20220301.de") #28GB RAM Usage

# Convert the text to ASCII values
texts = dataset['train']['text'] # Extract the text
x = np.array([], dtype=np.uint8) # Initialize the array

for text in texts:
  ascii_text = np.array([ord(c) for c in text], dtype=np.uint8) # Convert the string to ASCII values
  x = np.concatenate((x, ascii_text)) # Add the ASCII values to the array

  if len(x) >= int(95e6): # If the array has reached 95 million elements, break the loop
    x = x[:int(95e6)] # Truncate the array to 95 million elements

break

# Split the data into training and validation sets
train_x, valid_x = np.split(x, [int(0.9 * len(x))])
data_train, data_val = map(torch.from_numpy, (train_x, valid_x))

torch.save(model.state_dict(), 'megabyte-final.pt')

# CLI-Befehle
$ pip install MEGABYTE-pytorch datasets

$ python train.py

# Listing 2: Die deutsche WIkipedia in MEGABYTE einbinden
from MEGABYTE_pytorch import MEGABYTE
import torch

def decode_token(token):
  return str(chr(max(32, token)))

def decode_tokens(tokens):
  return ''.join(list(map(decode_token, tokens)))

def string_to_ascii(string):
  return [ord(c) for c in string]

model.eval()

input_str = 'Der Heise Verlag ist '

input_x = torch.tensor(string_to_ascii(input_str), dtype=torch.long).cuda()

with torch.no_grad():
  output = model.generate(input_x[None, :])
  output = output.flatten(1)
  output_str = decode_tokens(output[0][len(input_str):])
  print(output_str))

# Anzahl der trainierbaren Parameter ermitteln
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Das Modell hat {count_parameters(model)} Parameter.')

# Listing 3: MEGABYTE ein Prompt vervollständigen lassen
from MEGABYTE_pytorch import MEGABYTE
import torch

def decode_token(token):
  return str(chr(max(32, token)))

def decode_tokens(tokens):
  return ''.join(list(map(decode_token, tokens)))

def string_to_ascii(string):
  return [ord(c) for c in string]

model.eval()

input_str = 'Der Heise Verlag ist '

input_x = torch.tensor(string_to_ascii(input_str), dtype=torch.long).cuda()

with torch.no_grad():
  output = model.generate(input_x[None, :])
  output = output.flatten(1)
  output_str = decode_tokens(output[0][len(input_str):])
  print(output_str))

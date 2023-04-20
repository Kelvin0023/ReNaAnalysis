import torch
import pickle
import os

from renaanalysis.learning.HT import HierarchicalTransformer

# Load the model state_dict
model = HierarchicalTransformer() # replace YourModel with your actual model class
model.load_state_dict(torch.load(os.path.join('HT/RSVP-itemonset-locked', 'model.pt')))

# Load x_eeg
with open(os.path.join('HT/RSVP-itemonset-locked', 'x_eeg.pkl'), 'rb') as f:
    x_eeg = pickle.load(f)

# Load y
with open(os.path.join('HT/RSVP-itemonset-locked', 'y.pkl'), 'rb') as f:
    y = pickle.load(f)

# Load label_encoder
with open(os.path.join('HT/RSVP-itemonset-locked', 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
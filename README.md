# Analogical Matching Network
A neural-based system for outputting analogical matchings

## Experiment Setup

To set up experiments

   1. sh setup_directories.sh
   2. pip3 install -r requirements.txt (ensures package requirements are met)

There is a pretrained model included in the models directory, thus training 
is not necessary

## Prior to Training

Before training, you should generate the synthetic dataset

      python3 generate_dataset.py

This also builds the graph encoder object that goes along with the synthetic
dataset, which is needed for training

## Training a Model

To train a model

      python3 train_model.py

all results and intermediate progress is going to be stored in ./results, so
you should reference the files in there to see how your model is progressing
as it trains. All models are stored in ./models, with latest_model_obj.pkl being
the model called in the testing and visualization code

The default settings are set to those listed in the paper.

## Testing a Model

To test a model

      python3 test_model.py

test results are stored in ./results

## Visualizing a Model

To visualize outputs

      python3 visualize.py --domain <domain_name>

where <domain_name> is the domain you want to visualize analogies for, 
e.g. --domain atom. The 'custom' domain allows you to enter your own
base and target logical expression sets. An example of its use is

     python3 visualize.py --domain custom

     Please enter expressions for the Base at the prompt...
     Type END to finish...
     >>> (p (f a b))
     >>> (p (f c d))
     >>> END

     Please enter expressions for the Target at the prompt...
     Type END to finish...
     >>> (p (f b a))
     >>> (p (f d c))
     >>> END

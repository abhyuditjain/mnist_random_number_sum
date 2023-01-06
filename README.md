# What does it do?

It is a neural network that takes the following as inputs:

- Takes an MNIST image
- Takes a random number

And produces the following as outputs:

- Prediction of the number in the image
- Prediction of the sum of the number in the image and the random number provided

# How does it work?

Activation function used is ReLU everywhere.
For image prediction, it has 6 convolution layers and 3 FC layers. I do max pooling 2 times - after 5th and 6th convolution layers.

```
CONV 1    |>  28x28x1   | 26x26x32  | RF = 3x3
CONV 2    |>  26x26x32  | 24x24x64  | RF = 5x5
CONV 3    |>  24x24x64  | 22x22x128 | RF = 7x7
CONV 4    |>  22x22x128 | 20x20x256 | RF = 9x9
CONV 5    |>  20x20x256 | 18x18x512 | RF = 11x11
MAXPOOL   |>  18x18x512 | 9x9x512   | RF = 22x22
CONV 6    |>   9x9x512  | 7x7x1024  | RF = 24x24
MAXPOOL   |>   7x7x1024 | 3x3x1024  | RF = 48x48

FC 1      |> 1024*3*3   | 120
FC 2      |>      120   |  60
OUTPUT 1  |>       60   |  10        [These are the 10 classes from 0 to 9]
```

For sum prediction, I use 3 FC layers. For the 1st layer I concat the output from image prediction and the one-hot encoded random number, which results in 20 input features.

For example, if the random number is 4, it would be represented as `tensor([0 0 0 0 1 0 0 0 0 0])` This is concatenated to the `OUTPUT 1` to make it `[20]`

So, the layer at which the inputs are combined is labelled as `FC 3`

```
FC 3      |>       20   | 120
FC 4      |>      120   |  60
OUTPUT 2  |>       60   |  19        [These are the 19 classes from 0 to 18]
```

# Logs

```
EPOCH = 1 Image loss=2.300691604614258 + Sum loss=2.4757590293884277 for batch_id=468: 100%|██████████████████████████████████████████████████████████████| 469/469 [00:26<00:00, 17.83it/s]
Test set: Average loss (images): 4.8365, Accuracy: 1028/10000 (10%)
Test set: Average loss (sums): 4.8365, Accuracy: 972/10000 (10%)

EPOCH = 2 Image loss=0.12132460623979568 + Sum loss=1.3236684799194336 for batch_id=468: 100%|████████████████████████████████████████████████████████████| 469/469 [00:25<00:00, 18.53it/s]
Test set: Average loss (images): 1.4932, Accuracy: 9552/10000 (96%)
Test set: Average loss (sums): 1.4932, Accuracy: 4624/10000 (46%)

EPOCH = 3 Image loss=0.18859010934829712 + Sum loss=0.7601494193077087 for batch_id=468: 100%|████████████████████████████████████████████████████████████| 469/469 [00:25<00:00, 18.48it/s]
Test set: Average loss (images): 0.6403, Accuracy: 9748/10000 (97%)
Test set: Average loss (sums): 0.6403, Accuracy: 9060/10000 (91%)

EPOCH = 4 Image loss=0.032056067138910294 + Sum loss=0.13076896965503693 for batch_id=468: 100%|██████████████████████████████████████████████████████████| 469/469 [00:25<00:00, 18.45it/s]
Test set: Average loss (images): 0.2077, Accuracy: 9854/10000 (99%)
Test set: Average loss (sums): 0.2077, Accuracy: 9791/10000 (98%)

EPOCH = 5 Image loss=0.03335687145590782 + Sum loss=0.07385717332363129 for batch_id=468: 100%|███████████████████████████████████████████████████████████| 469/469 [00:25<00:00, 18.40it/s]
Test set: Average loss (images): 0.1319, Accuracy: 9879/10000 (99%)
Test set: Average loss (sums): 0.1319, Accuracy: 9851/10000 (99%)

EPOCH = 6 Image loss=0.018193380907177925 + Sum loss=0.0352884978055954 for batch_id=468: 100%|███████████████████████████████████████████████████████████| 469/469 [00:25<00:00, 18.40it/s]
Test set: Average loss (images): 0.1229, Accuracy: 9874/10000 (99%)
Test set: Average loss (sums): 0.1229, Accuracy: 9852/10000 (99%)

EPOCH = 7 Image loss=0.004262531641870737 + Sum loss=0.0151412608101964 for batch_id=468: 100%|███████████████████████████████████████████████████████████| 469/469 [00:25<00:00, 18.44it/s]
Test set: Average loss (images): 0.0871, Accuracy: 9917/10000 (99%)
Test set: Average loss (sums): 0.0871, Accuracy: 9889/10000 (99%)

EPOCH = 8 Image loss=0.033101752400398254 + Sum loss=0.07990437000989914 for batch_id=468: 100%|██████████████████████████████████████████████████████████| 469/469 [00:25<00:00, 18.39it/s]
Test set: Average loss (images): 0.0768, Accuracy: 9921/10000 (99%)
Test set: Average loss (sums): 0.0768, Accuracy: 9903/10000 (99%)

EPOCH = 9 Image loss=0.003930165898054838 + Sum loss=0.005559814628213644 for batch_id=468: 100%|█████████████████████████████████████████████████████████| 469/469 [00:25<00:00, 18.39it/s]
Test set: Average loss (images): 0.0821, Accuracy: 9909/10000 (99%)
Test set: Average loss (sums): 0.0821, Accuracy: 9902/10000 (99%)

EPOCH = 10 Image loss=0.00032622614526189864 + Sum loss=0.004662145860493183 for batch_id=468: 100%|██████████████████████████████████████████████████████| 469/469 [00:25<00:00, 18.39it/s]
Test set: Average loss (images): 0.0839, Accuracy: 9917/10000 (99%)
Test set: Average loss (sums): 0.0839, Accuracy: 9918/10000 (99%)

```

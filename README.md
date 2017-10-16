# rbf_tests

Test of adding RBF layer to learned model. Requires [rbf_keras](https://github.com/PetraVidnerova/rbf_keras).

## Usage:

```
usage: mnist_add_rbf.py [-h] [--betas BETAS] [--cnn] input output

positional arguments:
  input          input model saved in input.json and input_weights.h5
  output         output model saved in output.json and output_weights.h5

optional arguments:
  -h, --help     show this help message and exit
  --betas BETAS  initial value for betas
  --cnn          cnn type network (2d input)
```
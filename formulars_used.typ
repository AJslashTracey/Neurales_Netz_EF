#title[Formulars being used inside of our Project]


He weight initialization
$ W ~ cal(N)(0, 1) dot sqrt(2 / "fan_in") $

Linear layer (affine transform)
$ z^((l)) = a^((l - 1)) W^((l)) + b^((l)) $

ReLU activation (hidden layers)
$ a^((l)) = max(0, z^((l))) $

Softmax (output layer)
$ hat(y)_i = (e^(z_i)) / (sum_j e^(z_j)) $

Cross-entropy loss (one-hot labels)
$ cal(L)_C E = -1/m sum_(k=1)^m sum_c y_(k,c) log(hat(y)_(k,c)) $

L2 regularization (weight decay) added to loss
$ cal(L) = cal(L)_C E + (lambda / 2) sum_l norm(norm(W^((l))))^2 $

Output gradient (softmax + C E simplification)
$ d z^((L)) = hat(y) - y $

Gradients for each layer
$ d W^((l)) = ((a^((l - 1)))^T d z^((l))) / m, quad d b^((l)) = 1/m sum d z^((l)) $

Backprop through hidden ReLU layers
$ d z^((l)) = (d z^((l + 1)) (W^((l + 1)))^T) dot ReLU'(z^((l))) $

L2 term in gradients
$ d W^((l)) arrow.l d W^((l)) + lambda W^((l)) $

Gradient clipping (elementwise)
$ g arrow.l clip(g, -c, c) $

Gradient descent update
$ W^((l)) arrow.l W^((l)) - eta d W^((l)), quad b^((l)) arrow.l b^((l)) - eta d b^((l)) $

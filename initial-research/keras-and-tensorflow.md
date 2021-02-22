# keras & tensorflow

## main goals of this research

- what is tensorflow?
- what is keras?
- how do the two interact?
- what makes tensorflow different from pytorch?
- how do we get started doing the basic things with it/them?

## what is tensorflow

- free + open-source machine learning library
- developed by google
- written in python, c++, cuda
- cross platform: any desktop os, mobiles, server clusters, etc.
- a computation in tf is a stateful dataflow graph, where the data is a tensor
  (multi-d matrix). it's the *flow* of *tensors*. eh?
- google have made a tensor processing unit (tpu), which is specifically for
  things like maching learning and tf. has very good performance at running ml
  models
- stable python and c bindings
- not-as-stable bindings for c++, go, java, javascript
- 3rd-party bindings for c#, haskell, julia, matlab, r, scala, rust, ocaml, and
  crystal

## what is keras

- specifically to do with nn
- python based
- design goals
  - made for quick experimentation
  - user-friendly
  - modular
  - extensible
- contains a lot of the basic building blocks of nn
  - layers
  - objectives
  - activation functions
  - optimizers
  - image and text manipulation tools
  - dropout
  - batch normalization
  - pooling
- can produce models for lots

## how do keras and tensorflow interact

- keras is kind of the default for using tf
- interface for tensorflow - uses it as a back-end
- to do development with tf. you'll likely be using keras to develop your tf
  code - or if you're not, you'll spend lots of time reimplementing things

## tensorflow vs. pytorch

similarities

- machine learning libraries
- main bindings in python (thought other languages can also be used)
- high-level interfaces for creating machine learning models, particularly
  neural networks

differences

- facebook vs. google

## getting started with keras and tensorflow

both [keras](https://keras.io) and [tensorflow](https://www.tensorflow.org/)
have good documentation and tutorials on their websites.

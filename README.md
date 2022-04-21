
# FLINNG

Filters to Identify Near-Neighbor Groups (FLINNG) is a near neighbor search algorithm outlined in the paper 
[Practical Near Neighbor Search via Group Testing](https://arxiv.org/pdf/2106.11565.pdf). 

This branch (the main branch) contains a moderately cleaned up version of FLINNG. To 
access the original research code, use the research branch. Only this branch will 
be actively updated.

## Features

- C++ library built using CMAKE 
- Incremental/streaming index construction
- Parallel index construction and querying
- support for distance metrics I.P. and L2
- Index dumping to and from disk
- Improved API to support adding metadata and labels 

Note that some features of the research branch have yet to be ported over, and there are a few improvements
this branch might soon receive:
- densified minhash performance optimization (sparsify SRP, improve DOPH)

## Overview and Dataset Expectation

Please refer to the [Wiki Page](https://www.github.com/tonyzhang617/FLINNG/wiki) for a short overview of the algorithm.
The library expects Dataset to be normalized (ranges 0..1) for all data points. Support of arbitrary datasets will be added in future 
 

## Installation

To install library, run
```
git clone https://github.com/tonyzhang617/FLINNG.git
cd FLINNG/
mkdir build
cd build
cmake ..
sudo make install

```

## Authors
Implementation by [Josh Engels](https://www.github.com/joshengels) , [Tianyi (Tony) Zhang](https://www.github.com/tonyzhang617) and [Sameh Gobriel](https://www.github.com/s-gobriel). 
FLINNG created in collaboration with [Ben Coleman](https://randorithms.com/about.html)
and [Anshumali Shrivastava](https://www.cs.rice.edu/~as143/).

Please feel free to contact josh.adam.engels@gmail.com with any questions.


## Contributing

Currently, contributions are limited to bug fixes and suggestions. 
For a bug fix, feel free to submit a PR or send an email. 

Work on a production, supported version with efficient hash functions and an extended
feature set is ongoing at [ThirdAI](thirdai.com).

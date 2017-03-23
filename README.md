Legacy vectors
==============

Scripts to run neural nets over FS legacy vectors

* First, install tensorflow: https://www.tensorflow.org/install/install_mac
 (when you follow the instructions to run virtualenv to create your virtual environtment,
 add --python=python3.6 or something to get python3: `virtualenv --python=python3.6 tensorflow`)
* Second, download legacy-vectors.csv and put it in the "data" directory
* Third, run `pip3 install -r requirements.txt`

Single-layer
------------
* cd into the legacy/legacy directory and run `python3 single-layer.py`
 (the warnings are really unfortunate but it appears that you can ignore them.
 I didn't spend too much time trying to get rid of them since going forward I'm expecting we'll
 be coding directly to tensorflow core and not using tf.contrib.learn)

Running with 100,000 iterations should result in an accuracy (% examples labeled correctly) of .973.
This took 10 hours on my mac.

Two-layer
---------
* run `python3 two-layer.py`

Running with 500 iterations should result in an accuracy of .98. After that it starts to overfit.
There are things we can do to prevent overfitting, but I haven't implemented them yet.
This took 10 minutes on my mac

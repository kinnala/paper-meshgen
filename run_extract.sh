#!/bin/bash
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=600 --execute $1.ipynb
cat $1.nbconvert.ipynb | grep image/png | sed -e 's/"image\/png": "\(.*\)\\n",/\1/g' | awk '{print "echo \"" $1 "\" | base64 -d > image_" NR ".png"}' | xargs -I {} sh -c "{}"

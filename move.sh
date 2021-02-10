#!/bin/bash

for f in *.hdf5 do;
 mv "${f}" "data/logs/cp_150_epoch/"
done

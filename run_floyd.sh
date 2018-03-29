#!/bin/bash
floyd run --gpu --env pytorch-0.2 --mode serve --data taldehyde/datasets/deblurgan_weights:/checkpoints

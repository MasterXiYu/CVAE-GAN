#!/bin/bash
# ffmpeg -i ~/yixu_project/CVAE-GAN/output_VAE/images_epoch%02d_batch001.jpg -c:v libx264 -vf "fps=30,format=yuv420p" ~/yixu_project/CVAE-GAN/output_VAE/output_VAE.mp4
# ffmpeg -i ~/yixu_project/CVAE-GAN/output/images_epoch%02d_batch001.jpg -c:v libx264 -vf "fps=30,format=yuv420p" ~/yixu_project/CVAE-GAN/output/output_GAN.mp4
ffmpeg -i ~/yixu_project/CVAE-GAN/output_GAN_VAE/images_epoch%02d_batch001.jpg -c:v libx264 -vf "fps=30,format=yuv420p" ~/yixu_project/CVAE-GAN/output_GAN_VAE/output_GAN.mp4
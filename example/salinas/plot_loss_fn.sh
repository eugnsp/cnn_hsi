#!/usr/bin/gnuplot

set terminal pngcairo size 850,300 
set output 'salinas_loss_fn.png'
set multiplot layout 1,1
set xrange [0:1500]

set xlabel "Training iterations"
set ylabel "Loss function"
plot 'loss_fn.txt' with lines lw 3 notitle

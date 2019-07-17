#!/usr/bin/gnuplot

set terminal pngcairo size 2000,550 
set output 'salinas.png'
set multiplot layout 1,4
set pm3d map
set cbrange [0:]
set xrange [1:86]
set yrange [1:83]
unset cbtics
unset colorbox
unset border
unset xtics
unset ytics
set size square
set palette rgb -8,-3,-4

set xlabel "Groundtruth" font ",27"
plot 'gt.txt' matrix with image notitle

set xlabel "500 iterations, 78%"
plot '500.txt' matrix with image notitle

set xlabel "1000 iterations, 96%"
plot '1000.txt' matrix with image notitle

set xlabel "1500 iterations, 98%"
plot '1500.txt' matrix with image notitle

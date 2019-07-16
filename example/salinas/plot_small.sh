#!/usr/bin/gnuplot

set terminal pngcairo size 850,250 
set output 'salinas_sm.png'
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
set palette rgb -3,-8,-5

set xlabel "Groundtruth"
plot 'gt.txt' matrix with image notitle

set xlabel "500 iterations, 73%"
plot '500.txt' matrix with image notitle

set xlabel "1000 iterations, 87%"
plot '1000.txt' matrix with image notitle

set xlabel "1500 iterations, 91%"
plot '1500.txt' matrix with image notitle

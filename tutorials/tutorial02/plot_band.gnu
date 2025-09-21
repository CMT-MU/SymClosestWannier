unset key 
unset grid 
lwidth = 3 
set xrange [:0.6477783446554766] 
set yrange [-9.857128312527719:13.868896024856074] 
set tics font 'Times New Roman, 30' 

set size ratio 0.7 

set arrow from  0,  -9.857128312527719 to 0, 13.868896024856074 nohead 
set arrow from  0.2737823408618087,  -9.857128312527719 to 0.2737823408618087, 13.868896024856074 nohead 
set arrow from  0.41067351131324786,  -9.857128312527719 to 0.41067351131324786, 13.868896024856074 nohead 
set arrow from  0.6477783446554766,  -9.857128312527719 to 0.6477783446554766, 13.868896024856074 nohead 
set xtics ('{/Symbol G}' 0,'K' 0.2737823408618087,'M' 0.41067351131324786,'{/Symbol G}' 0.6477783446554766,) 

ef = -0.1863 
a = 2.435000000005462 
set terminal postscript eps color enhanced 

set output 'graphene_band.eps' 

plot 'graphene.band.gnu' u ($1/a):($2-ef) w l lw lwidth lc 'dark-grey', 'graphene_band.txt' u 1:2 w l lw lwidth dt (3,1) lc 'salmon', 0.0 lw 0.5 dt (2,1) lc 'black' 

set terminal pdf 

set output 'graphene_band.pdf' 

plot 'graphene.band.gnu' u ($1/a):($2-ef) w l lw lwidth lc 'dark-grey', 'graphene_band.txt' u 1:2 w l lw lwidth dt (3,1) lc 'salmon', 0.0 lw 0.5 dt (2,1) lc 'black'
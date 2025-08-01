unset key 
unset grid 
lwidth = 3 
set xrange [:0.6477783446554795] 
set yrange [-9.957018144648602:13.517843913222245] 
set tics font 'Times Roman, 30' 

set size ratio 0.7 

set arrow from  0,  -9.957018144648602 to  0, 13.517843913222245 nohead 
set arrow from  0.2371048333422304,  -9.957018144648602 to  0.2371048333422304, 13.517843913222245 nohead 
set arrow from  0.37399600379366954,  -9.957018144648602 to  0.37399600379366954, 13.517843913222245 nohead 
set arrow from  0.6477783446554795,  -9.957018144648602 to  0.6477783446554795, 13.517843913222245 nohead 
set xtics ('{/Symbol G}' 0,'M' 0.2371048333422304,'K' 0.37399600379366954,'{/Symbol G}' 0.6477783446554795,) 

ef = 0.0 
a = 2.435000000005462 
set terminal postscript eps color enhanced 

set output 'graphene_band_detail.eps' 

plot for [j=2:6:3] 'graphene_band_detail.txt' u 1:j w l lw lwidth dt (3,1) lc 'salmon', 0.0 lw 0.5 dt (2,1) lc 'black' 

set terminal pdf 

set output 'graphene_band_detail.pdf' 

plot 'graphene_band_detail.txt' u 1:2 w l lw lwidth dt (3,1) lc 'salmon', 0.0 lw 0.5 dt (2,1) lc 'black'
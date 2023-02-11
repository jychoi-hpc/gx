<<<<<<< HEAD
The python script gx_geo.py can be used to generate an eik.out file with geometry coefficients for a Miller equilibrium.<br/>
The output equilibrium file can then be read by gx with igeo=1. 

To generate the geometry, specify the Miller parameters in your gx input file. For example, in an input file called gx.in, one might have

[Geometry] <br/>
 igeo = 1 <br/>
 geofile = "gx.eik.out"<br/>
 rhoc = 0.5<br/>
 shat = 0.8<br/>
 Rmaj = 2.77778<br/>
 R_geo = 2.77778<br/>
 shift = 0.0<br/>
 qinp = 1.4<br/>
 akappa = 1.0<br/>
 akappri = 0.0<br/>
 tri = 0.0<br/>
 tripri = 0.0<br/>
 betaprim = 0.0<br/>


Next, use the gx_geo.py script via
```
> python gx_geo.py gx.in
=======
The python script gx\_geo.py can be used to generate an eik.out file with geometry coefficients, which can be read by gx with igeo=1. 

To generate the geometry, specify the Miller parameters in your gx input file. For example, in an input file called gx.in, one might have  

```
[Geometry]  
 igeo = 1  
 geofile = "gx.eik.out"  
 rhoc = 0.5  
 shat = 0.8  
 Rmaj = 2.77778  
 R\_geo = 2.77778  
 shift = 0.0  
 qinp = 1.4  
 akappa = 1.0  
 akappri = 0.0  
 tri = 0.0  
 tripri = 0.0  
 betaprim = 0.0  
```

Next, use the gx\_geo.py script via

```
python gx\_geo.py gx.in
```
>>>>>>> remotes/origin/add_bpar

```
This will generate a file gx.eik.out with the geometry coefficients.

Now you can run gx via
```
<<<<<<< HEAD
> [/path/to/]gx gx.in

```
=======
[/path/to/]gx gx.in
```
>>>>>>> remotes/origin/add_bpar

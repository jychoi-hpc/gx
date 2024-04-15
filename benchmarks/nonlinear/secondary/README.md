This test runs a secondary instability calculation. For details, see Rogers, Dorland, & Kotschenreuther, PRL (2000).

To run the test, use
```
gx kh01.in; gx kh01a.in
```

The correct final output should be roughly
```
ky	kx		omega		gamma
0.0000	-0.0500		-0.000160	4.901835
0.0000	0.0000
0.0000	0.0500		0.000160	4.901835

0.1000	-0.0500		0.000164	4.901835
0.1000	0.0000		0.000000	0.000000
0.1000	0.0500		-0.000164	4.901835
```

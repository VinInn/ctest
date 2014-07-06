

//
// from http://mathworld.wolfram.com/ApolloniusProblem.html
//


a = 2*(x1-x2);
b = 2*(y1-y2);
c = 0;
d = x1*x1 - x2*x2 + y1*y1 - y2*y2;


ap = 2*x1;
bp = 2*y1;
cp = 2*tp;
dp = x1*x1 + y1*y1; 

d1 = (a*bp-b*ap);
d2 = (a*bp-ap*b);

f1 = bp*d -b*dp;
f2 = a*dp-ap*d;

x = (f1+b*cp*r)/d1;
y = (f2-a*cp*r)/d2;


x*x + y*y = r*r + 2*r*tp + tp*tp;



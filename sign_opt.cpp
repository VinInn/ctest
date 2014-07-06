inline double chsign(double x, double sign) {
  return sign<0? -x : x;
}


double what1(double x, double k, double c) {
  double s=1.;
  if (k<c) s=-1.;
  return c + chsign(x,s);
}

double foo(double x, double k, double c) {
  double s=1.;
  if (k<c) s=-1.;
  return c + s*x;
}


double bar(double x, double k, double c) {
  return c + ( (k<c) ? -1.*x : 1.*x);
}


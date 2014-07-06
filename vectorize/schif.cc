float  * __restrict__  x, * __restrict__ y;
void foo16() {
for (int i = 16; --i >= 0; x[i] = 0) {}
}


void bar16() {
for (int i = 0; i!=16; ++i) x[i] = 0;
}

void bar16c() {
for (int i = 0; i!=16; ++i) x[i] = y[i];
}

void bar24() {
for (int i = 0; i!=24; ++i) x[i] = 0;
}

void bar24c() {
for (int i = 0; i!=24; ++i) x[i] = y[i];
}



void foo32() {
for (int i = 32; --i >= 0; x[i] = 0) {}
}


void bar32() {
for (int i = 0; i!=32; ++i) x[i] = 0;
}

void bar32_1() {
for (int i = 0; i!=32; ++i) x[i] = 1.f;
}

void bar32c() {
for (int i = 0; i!=32; ++i) x[i] = y[i];
}



	.cpu generic+fp+simd
	.file	"matmul.cc"
	.text
	.align	2
	.global	_Z6matmulv
	.type	_Z6matmulv, %function
_Z6matmulv:
.LFB0:
	adrp	x1, .LANCHOR0
	sub	sp, sp, #64
.LCFI0:
	add	x0, x1, :lo12:.LANCHOR0
	ldr	w13, [x1, #:lo12:.LANCHOR0]
	ldr	w17, [x0, 16]
	ldr	w16, [x0, 20]
	ldr	w15, [x0, 24]
	ldr	w14, [x0, 28]
	ldr	w12, [x0, 4]
	ldr	w11, [x0, 8]
	ldr	w10, [x0, 12]
	ldr	w9, [x0, 32]
	ldr	w8, [x0, 36]
	ldr	w7, [x0, 40]
	ldr	w6, [x0, 44]
	ldr	w5, [x0, 48]
	ldr	w4, [x0, 52]
	ldr	w3, [x0, 56]
	str	w17, [sp, 48]
	str	w16, [sp, 52]
	str	w15, [sp, 56]
	str	w14, [sp, 60]
	ldr	q17, [sp, 48]
	str	w13, [sp, 32]
	str	w12, [sp, 36]
	str	w11, [sp, 40]
	str	w10, [sp, 44]
	ldr	q18, [sp, 32]
	str	w9, [sp, 16]
	str	w8, [sp, 20]
	str	w7, [sp, 24]
	str	w6, [sp, 28]
	ldr	q0, [sp, 16]
	str	w5, [sp]
	str	w4, [sp, 4]
	str	w3, [sp, 8]
	ldr	w3, [x0, 60]
	ldr	q16, [x0, 64]
	ldr	q7, [x0, 80]
	ldr	q6, [x0, 96]
	ldr	q5, [x0, 112]
	str	w3, [sp, 12]
	ldr	q4, [sp]
	fmul	v3.4s, v0.4s, v16.s[2]
	add	sp, sp, 64
.LCFI1:
	fmul	v2.4s, v0.4s, v7.s[2]
	fmul	v1.4s, v0.4s, v6.s[2]
	fmul	v0.4s, v0.4s, v5.s[2]
	fmla	v3.4s, v18.4s, v16.4s[0]
	fmla	v2.4s, v18.4s, v7.4s[0]
	fmla	v1.4s, v18.4s, v6.4s[0]
	fmla	v0.4s, v18.4s, v5.4s[0]
	fmla	v3.4s, v17.4s, v16.4s[1]
	fmla	v2.4s, v17.4s, v7.4s[1]
	fmla	v1.4s, v17.4s, v6.4s[1]
	fmla	v0.4s, v17.4s, v5.4s[1]
	fmla	v3.4s, v4.4s, v16.4s[3]
	fmla	v2.4s, v4.4s, v7.4s[3]
	fmla	v1.4s, v4.4s, v6.4s[3]
	fmla	v0.4s, v4.4s, v5.4s[3]
	str	q3, [x0, 128]
	str	q2, [x0, 144]
	str	q1, [x0, 160]
	str	q0, [x0, 176]
	ret
.LFE0:
	.size	_Z6matmulv, .-_Z6matmulv
	.align	2
	.global	_Z7matmulUv
	.type	_Z7matmulUv, %function
_Z7matmulUv:
.LFB1:
	sub	sp, sp, #32
.LCFI2:
	adrp	x3, .LANCHOR0
	add	x3, x3, :lo12:.LANCHOR0
	stp	d8, d9, [sp]
	str	d10, [sp, 16]
.LCFI3:
	ldr	s7, [x3, 196]
	ldr	s21, [x3, 284]
	ldr	s6, [x3, 212]
	ldr	s31, [x3, 272]
	ldr	s29, [x3, 276]
	ldr	s27, [x3, 280]
	fmul	s5, s31, s7
	fmul	s4, s29, s7
	fmul	s3, s27, s7
	fmul	s9, s21, s7
	fmul	s2, s6, s31
	fmul	s1, s6, s29
	fmul	s0, s6, s27
	fmul	s20, s6, s21
	ldr	s7, [x3, 192]
	ldr	s8, [x3, 268]
	ldr	s6, [x3, 208]
	ldr	s24, [x3, 256]
	ldr	s23, [x3, 260]
	ldr	s22, [x3, 264]
	fmadd	s5, s24, s7, s5
	fmadd	s4, s23, s7, s4
	fmadd	s3, s22, s7, s3
	fmadd	s9, s8, s7, s9
	fmadd	s2, s6, s24, s2
	fmadd	s20, s6, s8, s20
	fmadd	s1, s6, s23, s1
	fmadd	s0, s6, s22, s0
	ldr	s7, [x3, 200]
	ldr	s6, [x3, 216]
	ldr	s19, [x3, 288]
	ldr	s18, [x3, 292]
	ldr	s17, [x3, 296]
	ldr	s30, [x3, 300]
	fmadd	s5, s19, s7, s5
	fmadd	s4, s18, s7, s4
	fmadd	s3, s17, s7, s3
	fmadd	s9, s30, s7, s9
	fmadd	s2, s6, s19, s2
	fmadd	s10, s6, s30, s20
	fmadd	s1, s6, s18, s1
	fmadd	s0, s6, s17, s0
	ldr	s25, [x3, 204]
	ldr	s16, [x3, 304]
	ldr	s7, [x3, 308]
	ldr	s6, [x3, 312]
	ldr	s28, [x3, 316]
	ldr	s20, [x3, 220]
	fmadd	s5, s16, s25, s5
	fmadd	s4, s7, s25, s4
	fmadd	s3, s6, s25, s3
	fmadd	s9, s28, s25, s9
	fmadd	s2, s20, s16, s2
	fmadd	s1, s20, s7, s1
	fmadd	s0, s20, s6, s0
	fmadd	s20, s20, s28, s10
	str	s5, [x3, 320]
	str	s4, [x3, 324]
	str	s3, [x3, 328]
	str	s9, [x3, 332]
	str	s2, [x3, 336]
	ldr	s25, [x3, 224]
	str	s1, [x3, 340]
	ldr	s26, [x3, 244]
	str	s20, [x3, 348]
	ldr	s20, [x3, 228]
	str	s0, [x3, 344]
	fmul	s2, s20, s31
	fmul	s1, s20, s29
	fmul	s0, s20, s27
	fmul	s31, s26, s31
	fmul	s29, s26, s29
	fmul	s27, s26, s27
	fmul	s20, s20, s21
	fmul	s26, s26, s21
	ldr	s21, [x3, 240]
	fmadd	s2, s25, s24, s2
	fmadd	s1, s25, s23, s1
	fmadd	s0, s25, s22, s0
	fmadd	s20, s25, s8, s20
	fmadd	s24, s21, s24, s31
	fmadd	s23, s21, s23, s29
	fmadd	s22, s21, s22, s27
	fmadd	s21, s21, s8, s26
	ldr	s4, [x3, 232]
	ldr	s5, [x3, 248]
	fmadd	s2, s4, s19, s2
	fmadd	s1, s4, s18, s1
	fmadd	s0, s4, s17, s0
	fmadd	s20, s4, s30, s20
	fmadd	s19, s5, s19, s24
	fmadd	s18, s5, s18, s23
	fmadd	s17, s5, s17, s22
	fmadd	s5, s5, s30, s21
	ldr	s4, [x3, 236]
	ldr	s3, [x3, 252]
	fmadd	s2, s4, s16, s2
	fmadd	s1, s4, s7, s1
	fmadd	s0, s4, s6, s0
	fmadd	s16, s3, s16, s19
	fmadd	s7, s3, s7, s18
	fmadd	s6, s3, s6, s17
	fmadd	s4, s4, s28, s20
	fmadd	s3, s3, s28, s5
	str	s2, [x3, 352]
	str	s1, [x3, 356]
	str	s0, [x3, 360]
	str	s16, [x3, 368]
	str	s7, [x3, 372]
	str	s6, [x3, 376]
	str	s4, [x3, 364]
	str	s3, [x3, 380]
	ldp	d8, d9, [sp]
.LCFI4:
	ldr	d10, [sp, 16]
.LCFI5:
	add	sp, sp, 32
.LCFI6:
	ret
.LFE1:
	.size	_Z7matmulUv, .-_Z7matmulUv
	.global	dest
	.global	src2
	.global	src1
	.global	c
	.global	b
	.global	a
	.bss
	.align	5
.LANCHOR0 = . + 0
	.type	b, %object
	.size	b, 64
b:
	.zero	64
	.type	a, %object
	.size	a, 64
a:
	.zero	64
	.type	c, %object
	.size	c, 64
c:
	.zero	64
	.type	src1, %object
	.size	src1, 64
src1:
	.zero	64
	.type	src2, %object
	.size	src2, 64
src2:
	.zero	64
	.type	dest, %object
	.size	dest, 64
dest:
	.zero	64
	.ident	"GCC: (GNU) 4.10.0 20140705 (experimental) [trunk revision 212302]"

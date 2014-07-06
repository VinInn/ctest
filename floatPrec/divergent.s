	.text
	.align 4,0x90
	.globl __Z7computev
__Z7computev:
LFB221:
	vmovaps	LC0(%rip), %ymm7
	leaq	_a(%rip), %rsi
	xorl	%eax, %eax
	vmovaps	LC1(%rip), %ymm6
	leaq	_b(%rip), %rcx
	vxorps	%xmm5, %xmm5, %xmm5
	leaq	_c(%rip), %rdx
	.align 4,0x90
L3:
	vmovaps	(%rsi,%rax), %ymm2
	vmovaps	(%rcx,%rax), %ymm1
	vsubps	%ymm1, %ymm2, %ymm3
	vcmpneqps	%ymm3, %ymm5, %ymm4
	vrsqrtps	%ymm3, %ymm0
	vandps	%ymm4, %ymm0, %ymm0
	vmulps	%ymm3, %ymm0, %ymm3
	vmulps	%ymm0, %ymm3, %ymm0
	vaddps	%ymm7, %ymm0, %ymm0
	vmulps	%ymm6, %ymm3, %ymm3
	vmulps	%ymm3, %ymm0, %ymm3
	vaddps	%ymm2, %ymm1, %ymm0
	vcmpltps	%ymm1, %ymm2, %ymm1
	vblendvps	%ymm1, %ymm0, %ymm3, %ymm1
	vmovaps	%ymm1, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L3
	vzeroupper
	ret
LFE221:
	.align 4,0x90
	.globl __Z8computeBv
__Z8computeBv:
LFB222:
	pushq	%rbp
LCFI0:
	leaq	_a(%rip), %rcx
	leaq	_b(%rip), %rsi
	movq	%rsp, %rbp
LCFI1:
	andq	$-32, %rsp
	subq	$904, %rsp
	movq	%rcx, %rax
	movq	%rsi, %r9
	leaq	-120(%rsp), %rdx
	leaq	_c(%rip), %rdi
	movq	%rdx, %r10
	leaq	4096+_a(%rip), %r11
	movq	%rdi, %r8
	vmovdqa	LC2(%rip), %ymm0
	vmovdqa	LC3(%rip), %ymm10
	vmovdqa	LC4(%rip), %ymm9
	vmovdqa	LC5(%rip), %ymm12
	vmovdqa	LC6(%rip), %ymm11
L7:
	vmovaps	(%rax), %ymm7
	subq	$-128, %rax
	subq	$-128, %r9
	vmovaps	-96(%rax), %ymm5
	addq	$32, %r10
	subq	$-128, %r8
	vmovaps	-128(%r9), %ymm8
	vmovaps	-96(%r9), %ymm6
	vcmpltps	%ymm8, %ymm7, %ymm14
	vmovaps	-64(%rax), %ymm3
	vcmpltps	%ymm6, %ymm5, %ymm13
	vmovaps	-32(%rax), %ymm1
	vmovaps	-64(%r9), %ymm4
	vaddps	%ymm7, %ymm8, %ymm7
	vmovaps	-32(%r9), %ymm2
	vpand	%ymm14, %ymm0, %ymm14
	vaddps	%ymm5, %ymm6, %ymm5
	vcmpltps	%ymm4, %ymm3, %ymm15
	vpand	%ymm13, %ymm0, %ymm13
	vpshufb	%ymm10, %ymm14, %ymm14
	vpshufb	%ymm9, %ymm13, %ymm13
	vaddps	%ymm3, %ymm4, %ymm3
	vpor	%ymm13, %ymm14, %ymm14
	vcmpltps	%ymm2, %ymm1, %ymm13
	vpermq	$216, %ymm14, %ymm14
	vpshufb	%ymm12, %ymm14, %ymm14
	vaddps	%ymm1, %ymm2, %ymm1
	vpand	%ymm15, %ymm0, %ymm15
	vpshufb	%ymm10, %ymm15, %ymm15
	vpand	%ymm13, %ymm0, %ymm13
	vpshufb	%ymm9, %ymm13, %ymm13
	vpor	%ymm13, %ymm15, %ymm13
	vpermq	$216, %ymm13, %ymm13
	vpshufb	%ymm11, %ymm13, %ymm13
	vpor	%ymm13, %ymm14, %ymm13
	vpermq	$216, %ymm13, %ymm13
	vmovdqa	%ymm13, -32(%r10)
	vmovaps	%ymm7, -128(%r8)
	vmovaps	%ymm5, -96(%r8)
	vmovaps	%ymm3, -64(%r8)
	vmovaps	%ymm1, -32(%r8)
	cmpq	%r11, %rax
	jne	L7
	xorl	%eax, %eax
	.align 4,0x90
L10:
	cmpb	$0, (%rdx,%rax)
	je	L8
	vmovss	(%rcx,%rax,4), %xmm0
	vsubss	(%rsi,%rax,4), %xmm0, %xmm0
	vsqrtss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, (%rdi,%rax,4)
L8:
	addq	$1, %rax
	cmpq	$1024, %rax
	jne	L10
	vzeroupper
	leave
LCFI2:
	ret
LFE222:
	.align 4,0x90
	.globl __Z8computeVv
__Z8computeVv:
LFB1235:
	leaq	_va(%rip), %rsi
	xorl	%eax, %eax
	leaq	_vb(%rip), %rcx
	jmp	L16
	.align 4,0x90
L13:
	vsubps	%ymm0, %ymm1, %ymm0
	vblendvps	%ymm2, %ymm3, %ymm0, %ymm2
L14:
	leaq	_vc(%rip), %rdx
	vmovaps	%ymm2, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$32768, %rax
	je	L17
L16:
	vmovaps	(%rsi,%rax), %ymm1
	vmovaps	(%rcx,%rax), %ymm0
	vcmpltps	%ymm0, %ymm1, %ymm2
	vaddps	%ymm1, %ymm0, %ymm3
	vmovmskps	%ymm2, %edx
	cmpl	$255, %edx
	jne	L13
	vmovaps	%ymm3, %ymm2
	jmp	L14
	.align 4,0x90
L17:
	vzeroupper
	ret
LFE1235:
	.globl _vc
	.zerofill __DATA,__pu_bss5,_vc,32768,5
	.globl _vb
	.zerofill __DATA,__pu_bss5,_vb,32768,5
	.globl _va
	.zerofill __DATA,__pu_bss5,_va,32768,5
	.globl _c
	.zerofill __DATA,__pu_bss5,_c,4096,5
	.globl _b
	.zerofill __DATA,__pu_bss5,_b,4096,5
	.globl _a
	.zerofill __DATA,__pu_bss5,_a,4096,5
	.const
	.align 5
LC0:
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.align 5
LC1:
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.align 5
LC2:
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.align 5
LC3:
	.byte	0
	.byte	1
	.byte	4
	.byte	5
	.byte	8
	.byte	9
	.byte	12
	.byte	13
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	4
	.byte	5
	.byte	8
	.byte	9
	.byte	12
	.byte	13
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 5
LC4:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	4
	.byte	5
	.byte	8
	.byte	9
	.byte	12
	.byte	13
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	4
	.byte	5
	.byte	8
	.byte	9
	.byte	12
	.byte	13
	.align 5
LC5:
	.byte	0
	.byte	2
	.byte	4
	.byte	6
	.byte	8
	.byte	10
	.byte	12
	.byte	14
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	2
	.byte	4
	.byte	6
	.byte	8
	.byte	10
	.byte	12
	.byte	14
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 5
LC6:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	2
	.byte	4
	.byte	6
	.byte	8
	.byte	10
	.byte	12
	.byte	14
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	2
	.byte	4
	.byte	6
	.byte	8
	.byte	10
	.byte	12
	.byte	14
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x1
	.ascii "zR\0"
	.byte	0x1
	.byte	0x78
	.byte	0x10
	.byte	0x1
	.byte	0x10
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x90
	.byte	0x1
	.align 3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB221-.
	.set L$set$2,LFE221-LFB221
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB222-.
	.set L$set$4,LFE222-LFB222
	.quad L$set$4
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI0-LFB222
	.long L$set$5
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$6,LCFI1-LCFI0
	.long L$set$6
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$7,LCFI2-LCFI1
	.long L$set$7
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$8,LEFDE5-LASFDE5
	.long L$set$8
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB1235-.
	.set L$set$9,LFE1235-LFB1235
	.quad L$set$9
	.byte	0
	.align 3
LEFDE5:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

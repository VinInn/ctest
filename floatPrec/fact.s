	.text
	.align 4,0x90
	.globl __Z7computev
__Z7computev:
LFB224:
	vmovaps	LC0(%rip), %ymm10
	xorl	%eax, %eax
	vxorps	%xmm5, %xmm5, %xmm5
	vmovaps	LC1(%rip), %ymm9
	leaq	_a(%rip), %rcx
	vmovaps	LC2(%rip), %ymm8
	leaq	_b(%rip), %rdx
	vmovaps	LC3(%rip), %ymm7
	vmovaps	LC4(%rip), %ymm6
	.align 4,0x90
L3:
	vmovaps	(%rcx,%rax), %ymm2
	vmulps	%ymm2, %ymm2, %ymm0
	vcmpneqps	%ymm0, %ymm5, %ymm3
	vrsqrtps	%ymm0, %ymm1
	vsubps	%ymm7, %ymm0, %ymm4
	vandps	%ymm3, %ymm1, %ymm1
	vmulps	%ymm0, %ymm1, %ymm3
	vmulps	%ymm1, %ymm3, %ymm1
	vaddps	%ymm10, %ymm1, %ymm1
	vmulps	%ymm9, %ymm3, %ymm3
	vcmpltps	%ymm8, %ymm0, %ymm0
	vmulps	%ymm3, %ymm1, %ymm3
	vsubps	%ymm6, %ymm3, %ymm3
	vblendvps	%ymm0, %ymm4, %ymm3, %ymm0
	vmulps	%ymm2, %ymm0, %ymm2
	vmovaps	%ymm2, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$32768, %rax
	jne	L3
	vzeroupper
	ret
LFE224:
	.globl _b
	.zerofill __DATA,__pu_bss5,_b,32768,5
	.globl _a
	.zerofill __DATA,__pu_bss5,_a,32768,5
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
	.long	1084227584
	.long	1084227584
	.long	1084227584
	.long	1084227584
	.long	1084227584
	.long	1084227584
	.long	1084227584
	.long	1084227584
	.align 5
LC3:
	.long	1075838976
	.long	1075838976
	.long	1075838976
	.long	1075838976
	.long	1075838976
	.long	1075838976
	.long	1075838976
	.long	1075838976
	.align 5
LC4:
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
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
	.quad	LFB224-.
	.set L$set$2,LFE224-LFB224
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

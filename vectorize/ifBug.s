	.text
	.align 4,0x90
	.globl __Z5loop0v
__Z5loop0v:
LFB0:
	xorl	%eax, %eax
	movss	LC0(%rip), %xmm1
	leaq	_j1(%rip), %rcx
	leaq	_j2(%rip), %rdx
	.align 4,0x90
L3:
	movl	(%rdx,%rax), %esi
	cmpl	%esi, (%rcx,%rax)
	jge	L2
	leaq	_d(%rip), %rsi
	movss	(%rsi,%rax), %xmm0
	leaq	_c(%rip), %rsi
	xorps	%xmm1, %xmm0
	movss	%xmm0, (%rsi,%rax)
L2:
	addq	$4, %rax
	cmpq	$256, %rax
	jne	L3
	rep
	ret
LFE0:
	.align 4,0x90
	.globl __Z5loop1v
__Z5loop1v:
LFB1:
	xorl	%eax, %eax
	xorps	%xmm1, %xmm1
	movss	LC0(%rip), %xmm2
	leaq	_c(%rip), %rdx
	jmp	L9
	.align 4,0x90
L7:
	addq	$4, %rax
	cmpq	$256, %rax
	je	L12
L9:
	comiss	(%rdx,%rax), %xmm1
	jbe	L7
	leaq	_d(%rip), %rcx
	movss	(%rcx,%rax), %xmm0
	xorps	%xmm2, %xmm0
	movss	%xmm0, (%rcx,%rax)
	addq	$4, %rax
	cmpq	$256, %rax
	jne	L9
L12:
	rep
	ret
LFE1:
	.align 4,0x90
	.globl __Z5loop2v
__Z5loop2v:
LFB2:
	movaps	_c(%rip), %xmm0
	xorps	%xmm1, %xmm1
	movaps	_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	LC2(%rip), %xmm2
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	16+_c(%rip), %xmm0
	movaps	%xmm3, _d(%rip)
	movaps	16+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	32+_c(%rip), %xmm0
	movaps	%xmm3, 16+_d(%rip)
	movaps	32+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	48+_c(%rip), %xmm0
	movaps	%xmm3, 32+_d(%rip)
	movaps	48+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	64+_c(%rip), %xmm0
	movaps	%xmm3, 48+_d(%rip)
	movaps	64+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	80+_c(%rip), %xmm0
	movaps	%xmm3, 64+_d(%rip)
	movaps	80+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	96+_c(%rip), %xmm0
	movaps	%xmm3, 80+_d(%rip)
	movaps	96+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	112+_c(%rip), %xmm0
	movaps	%xmm3, 96+_d(%rip)
	movaps	112+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	128+_c(%rip), %xmm0
	movaps	%xmm3, 112+_d(%rip)
	movaps	128+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	144+_c(%rip), %xmm0
	movaps	%xmm3, 128+_d(%rip)
	movaps	144+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	160+_c(%rip), %xmm0
	movaps	%xmm3, 144+_d(%rip)
	movaps	160+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	%xmm3, 160+_d(%rip)
	movaps	176+_c(%rip), %xmm0
	movaps	176+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	192+_c(%rip), %xmm0
	movaps	%xmm3, 176+_d(%rip)
	movaps	192+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	208+_c(%rip), %xmm0
	movaps	%xmm3, 192+_d(%rip)
	movaps	208+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	224+_c(%rip), %xmm0
	movaps	%xmm3, 208+_d(%rip)
	movaps	224+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	movaps	%xmm3, %xmm4
	xorps	%xmm2, %xmm4
	blendvps	%xmm0, %xmm4, %xmm3
	movaps	240+_c(%rip), %xmm0
	movaps	%xmm3, 224+_d(%rip)
	movaps	240+_d(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	xorps	%xmm3, %xmm2
	blendvps	%xmm0, %xmm2, %xmm3
	movaps	%xmm3, 240+_d(%rip)
	ret
LFE2:
	.align 4,0x90
	.globl __Z5loop3v
__Z5loop3v:
LFB3:
	movdqa	_j2(%rip), %xmm0
	movaps	_d(%rip), %xmm2
	pcmpgtd	_j1(%rip), %xmm0
	movaps	LC2(%rip), %xmm1
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	16+_j2(%rip), %xmm0
	movaps	%xmm2, _d(%rip)
	movaps	16+_d(%rip), %xmm2
	pcmpgtd	16+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	32+_j2(%rip), %xmm0
	movaps	%xmm2, 16+_d(%rip)
	movaps	32+_d(%rip), %xmm2
	pcmpgtd	32+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	48+_j2(%rip), %xmm0
	movaps	%xmm2, 32+_d(%rip)
	movaps	48+_d(%rip), %xmm2
	pcmpgtd	48+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	64+_j2(%rip), %xmm0
	movaps	%xmm2, 48+_d(%rip)
	movaps	64+_d(%rip), %xmm2
	pcmpgtd	64+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	80+_j2(%rip), %xmm0
	movaps	%xmm2, 64+_d(%rip)
	movaps	80+_d(%rip), %xmm2
	pcmpgtd	80+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	96+_j2(%rip), %xmm0
	movaps	%xmm2, 80+_d(%rip)
	movaps	96+_d(%rip), %xmm2
	pcmpgtd	96+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	112+_j2(%rip), %xmm0
	movaps	%xmm2, 96+_d(%rip)
	movaps	112+_d(%rip), %xmm2
	pcmpgtd	112+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	128+_j2(%rip), %xmm0
	movaps	%xmm2, 112+_d(%rip)
	movaps	128+_d(%rip), %xmm2
	pcmpgtd	128+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movaps	%xmm2, 128+_d(%rip)
	movdqa	144+_j2(%rip), %xmm0
	movaps	144+_d(%rip), %xmm2
	pcmpgtd	144+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	160+_j2(%rip), %xmm0
	movaps	%xmm2, 144+_d(%rip)
	movaps	160+_d(%rip), %xmm2
	pcmpgtd	160+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	176+_j2(%rip), %xmm0
	movaps	%xmm2, 160+_d(%rip)
	movaps	176+_d(%rip), %xmm2
	pcmpgtd	176+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	192+_j2(%rip), %xmm0
	movaps	%xmm2, 176+_d(%rip)
	movaps	192+_d(%rip), %xmm2
	pcmpgtd	192+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	208+_j2(%rip), %xmm0
	movaps	%xmm2, 192+_d(%rip)
	movaps	208+_d(%rip), %xmm2
	pcmpgtd	208+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	224+_j2(%rip), %xmm0
	movaps	%xmm2, 208+_d(%rip)
	movaps	224+_d(%rip), %xmm2
	pcmpgtd	224+_j1(%rip), %xmm0
	movaps	%xmm2, %xmm3
	xorps	%xmm1, %xmm3
	blendvps	%xmm0, %xmm3, %xmm2
	movdqa	240+_j2(%rip), %xmm0
	movaps	%xmm2, 224+_d(%rip)
	movaps	240+_d(%rip), %xmm2
	pcmpgtd	240+_j1(%rip), %xmm0
	xorps	%xmm2, %xmm1
	blendvps	%xmm0, %xmm1, %xmm2
	movaps	%xmm2, 240+_d(%rip)
	ret
LFE3:
	.globl _j2
	.zerofill __DATA,__pu_bss5,_j2,256,5
	.globl _j1
	.zerofill __DATA,__pu_bss5,_j1,256,5
	.globl _d
	.zerofill __DATA,__pu_bss5,_d,256,5
	.globl _c
	.zerofill __DATA,__pu_bss5,_c,256,5
	.literal16
	.align 4
LC0:
	.long	2147483648
	.long	0
	.long	0
	.long	0
	.align 4
LC2:
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
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
	.quad	LFB0-.
	.set L$set$2,LFE0-LFB0
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1-.
	.set L$set$4,LFE1-LFB1
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB2-.
	.set L$set$6,LFE2-LFB2
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB3-.
	.set L$set$8,LFE3-LFB3
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

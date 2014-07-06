	.text
	.align 4,0x90
	.globl __Z1ff
__Z1ff:
LFB1906:
	comiss	_lim(%rip), %xmm0
	jb	L23
	comiss	4+_lim(%rip), %xmm0
	jb	L24
	comiss	8+_lim(%rip), %xmm0
	jb	L25
	comiss	12+_lim(%rip), %xmm0
	jb	L26
	comiss	16+_lim(%rip), %xmm0
	jb	L27
	jmp	__Z2f5f
	.align 4,0x90
L23:
	jmp	__Z2f0f
	.align 4,0x90
L25:
	jmp	__Z2f2f
	.align 4,0x90
L24:
	jmp	__Z2f1f
	.align 4,0x90
L27:
	jmp	__Z2f4f
	.align 4,0x90
L26:
	jmp	__Z2f3f
LFE1906:
	.align 4,0x90
	.globl __Z1fU8__vectorf
__Z1fU8__vectorf:
LFB1907:
	subq	$168, %rsp
LCFI0:
	movaps	%xmm0, %xmm5
	movaps	%xmm0, %xmm1
	movss	_lim(%rip), %xmm3
	movaps	%xmm1, %xmm4
	movaps	%xmm1, %xmm6
	shufps	$0, %xmm3, %xmm3
	cmpltps	%xmm3, %xmm5
	movaps	%xmm3, %xmm0
	movss	4+_lim(%rip), %xmm3
	cmpltps	%xmm1, %xmm0
	shufps	$0, %xmm3, %xmm3
	cmpltps	%xmm3, %xmm4
	movmskps	%xmm5, %eax
	cmpltps	%xmm1, %xmm3
	testl	%eax, %eax
	pand	%xmm4, %xmm0
	movss	8+_lim(%rip), %xmm4
	movaps	%xmm0, 80(%rsp)
	movaps	%xmm1, %xmm0
	shufps	$0, %xmm4, %xmm4
	cmpltps	%xmm4, %xmm0
	cmpltps	%xmm1, %xmm4
	pand	%xmm0, %xmm3
	movaps	%xmm3, 96(%rsp)
	movaps	%xmm1, %xmm0
	movss	12+_lim(%rip), %xmm3
	shufps	$0, %xmm3, %xmm3
	cmpltps	%xmm3, %xmm0
	cmpltps	%xmm1, %xmm3
	pand	%xmm0, %xmm4
	movss	16+_lim(%rip), %xmm0
	movaps	%xmm4, 112(%rsp)
	shufps	$0, %xmm0, %xmm0
	cmpltps	%xmm0, %xmm6
	pand	%xmm6, %xmm3
	movaps	%xmm3, 128(%rsp)
	movss	20+_lim(%rip), %xmm3
	shufps	$0, %xmm3, %xmm3
	cmpltps	%xmm1, %xmm3
	jne	L54
L29:
	movaps	80(%rsp), %xmm0
	movmskps	%xmm0, %eax
	testl	%eax, %eax
	jne	L55
L30:
	movaps	96(%rsp), %xmm0
	movmskps	%xmm0, %eax
	testl	%eax, %eax
	jne	L56
L31:
	movaps	112(%rsp), %xmm0
	movmskps	%xmm0, %eax
	testl	%eax, %eax
	jne	L57
L32:
	movaps	128(%rsp), %xmm0
	movmskps	%xmm0, %eax
	testl	%eax, %eax
	jne	L58
L33:
	movmskps	%xmm3, %eax
	testl	%eax, %eax
	je	L34
	movss	16+_lim(%rip), %xmm4
	movaps	%xmm1, %xmm0
	movaps	%xmm2, 32(%rsp)
	movaps	%xmm1, (%rsp)
	shufps	$0, %xmm4, %xmm4
	movaps	%xmm4, 16(%rsp)
	call	__Z2f5U8__vectorf
	movaps	16(%rsp), %xmm4
	movaps	(%rsp), %xmm1
	movaps	%xmm0, %xmm3
	movaps	32(%rsp), %xmm2
	cmpltps	%xmm1, %xmm4
	movaps	%xmm4, %xmm0
	blendvps	%xmm0, %xmm3, %xmm2
L34:
	movaps	%xmm2, %xmm0
	addq	$168, %rsp
LCFI1:
	ret
	.align 4,0x90
L58:
LCFI2:
	movss	16+_lim(%rip), %xmm4
	movaps	%xmm1, %xmm6
	movaps	%xmm3, 48(%rsp)
	movss	12+_lim(%rip), %xmm0
	movaps	%xmm2, 32(%rsp)
	shufps	$0, %xmm4, %xmm4
	cmpltps	%xmm4, %xmm6
	movaps	%xmm1, 16(%rsp)
	shufps	$0, %xmm0, %xmm0
	cmpltps	%xmm1, %xmm0
	movdqa	%xmm6, %xmm7
	pand	%xmm0, %xmm7
	movaps	%xmm1, %xmm0
	movdqa	%xmm7, (%rsp)
	call	__Z2f4U8__vectorf
	movaps	32(%rsp), %xmm2
	movaps	%xmm0, %xmm5
	pxor	%xmm0, %xmm0
	pcmpeqd	(%rsp), %xmm0
	movaps	48(%rsp), %xmm3
	movaps	16(%rsp), %xmm1
	blendvps	%xmm0, %xmm2, %xmm5
	movaps	%xmm5, %xmm2
	jmp	L33
	.align 4,0x90
L57:
	movss	12+_lim(%rip), %xmm4
	movaps	%xmm1, %xmm6
	movaps	%xmm3, 48(%rsp)
	movss	8+_lim(%rip), %xmm0
	movaps	%xmm2, 32(%rsp)
	shufps	$0, %xmm4, %xmm4
	cmpltps	%xmm4, %xmm6
	movaps	%xmm1, 16(%rsp)
	shufps	$0, %xmm0, %xmm0
	cmpltps	%xmm1, %xmm0
	movdqa	%xmm6, %xmm7
	pand	%xmm0, %xmm7
	movaps	%xmm1, %xmm0
	movdqa	%xmm7, (%rsp)
	call	__Z2f3U8__vectorf
	movaps	32(%rsp), %xmm2
	movaps	%xmm0, %xmm5
	pxor	%xmm0, %xmm0
	pcmpeqd	(%rsp), %xmm0
	movaps	48(%rsp), %xmm3
	movaps	16(%rsp), %xmm1
	blendvps	%xmm0, %xmm2, %xmm5
	movaps	%xmm5, %xmm2
	jmp	L32
	.align 4,0x90
L56:
	movss	4+_lim(%rip), %xmm0
	movaps	%xmm1, %xmm7
	movaps	%xmm3, 48(%rsp)
	movss	8+_lim(%rip), %xmm4
	movaps	%xmm2, 32(%rsp)
	shufps	$0, %xmm0, %xmm0
	cmpltps	%xmm1, %xmm0
	movaps	%xmm1, 16(%rsp)
	shufps	$0, %xmm4, %xmm4
	cmpltps	%xmm4, %xmm7
	pand	%xmm0, %xmm7
	movaps	%xmm1, %xmm0
	movdqa	%xmm7, (%rsp)
	call	__Z2f2U8__vectorf
	movaps	32(%rsp), %xmm2
	movaps	%xmm0, %xmm5
	pxor	%xmm0, %xmm0
	pcmpeqd	(%rsp), %xmm0
	movaps	48(%rsp), %xmm3
	movaps	16(%rsp), %xmm1
	blendvps	%xmm0, %xmm2, %xmm5
	movaps	%xmm5, %xmm2
	jmp	L31
	.align 4,0x90
L55:
	movss	_lim(%rip), %xmm0
	movaps	%xmm1, %xmm7
	movaps	%xmm3, 48(%rsp)
	movss	4+_lim(%rip), %xmm4
	movaps	%xmm2, 32(%rsp)
	shufps	$0, %xmm0, %xmm0
	cmpltps	%xmm1, %xmm0
	movaps	%xmm1, 16(%rsp)
	shufps	$0, %xmm4, %xmm4
	cmpltps	%xmm4, %xmm7
	pand	%xmm0, %xmm7
	movaps	%xmm1, %xmm0
	movdqa	%xmm7, (%rsp)
	call	__Z2f1U8__vectorf
	movaps	32(%rsp), %xmm2
	movaps	%xmm0, %xmm5
	pxor	%xmm0, %xmm0
	pcmpeqd	(%rsp), %xmm0
	movaps	48(%rsp), %xmm3
	movaps	16(%rsp), %xmm1
	blendvps	%xmm0, %xmm2, %xmm5
	movaps	%xmm5, %xmm2
	jmp	L30
	.align 4,0x90
L54:
	movaps	%xmm1, %xmm0
	movaps	%xmm3, 32(%rsp)
	movaps	%xmm5, 16(%rsp)
	movaps	%xmm1, (%rsp)
	call	__Z2f0U8__vectorf
	movaps	16(%rsp), %xmm5
	xorps	%xmm2, %xmm2
	movaps	%xmm0, %xmm4
	movaps	32(%rsp), %xmm3
	movaps	%xmm5, %xmm0
	movaps	(%rsp), %xmm1
	blendvps	%xmm0, %xmm4, %xmm2
	jmp	L29
LFE1907:
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
__GLOBAL__sub_I_VSwitch.cpp:
LFB2057:
	leaq	__ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
LCFI3:
	call	__ZNSt8ios_base4InitC1Ev
	movq	__ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	addq	$8, %rsp
LCFI4:
	leaq	___dso_handle(%rip), %rdx
	leaq	__ZStL8__ioinit(%rip), %rsi
	jmp	___cxa_atexit
LFE2057:
	.globl _lim
	.zerofill __DATA,__pu_bss4,_lim,20,4
	.static_data
__ZStL8__ioinit:
	.space	1
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
	.quad	LFB1906-.
	.set L$set$2,LFE1906-LFB1906
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1907-.
	.set L$set$4,LFE1907-LFB1907
	.quad L$set$4
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI0-LFB1907
	.long L$set$5
	.byte	0xe
	.byte	0xb0,0x1
	.byte	0x4
	.set L$set$6,LCFI1-LCFI0
	.long L$set$6
	.byte	0xa
	.byte	0xe
	.byte	0x8
	.byte	0x4
	.set L$set$7,LCFI2-LCFI1
	.long L$set$7
	.byte	0xb
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$8,LEFDE5-LASFDE5
	.long L$set$8
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB2057-.
	.set L$set$9,LFE2057-LFB2057
	.quad L$set$9
	.byte	0
	.byte	0x4
	.set L$set$10,LCFI3-LFB2057
	.long L$set$10
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$11,LCFI4-LCFI3
	.long L$set$11
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE5:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_VSwitch.cpp
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

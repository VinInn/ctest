	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB1:
	.text
LHOTB1:
	.align 4,0x90
	.globl __Z4barXv
__Z4barXv:
LFB0:
	movdqa	LC0(%rip), %xmm5
	xorl	%eax, %eax
	pxor	%xmm4, %xmm4
	pxor	%xmm3, %xmm3
	leaq	_y(%rip), %r8
	leaq	_x(%rip), %rdi
	leaq	_w(%rip), %rsi
	leaq	_k(%rip), %rcx
	leaq	_z(%rip), %rdx
	.align 4,0x90
L2:
	movaps	(%r8,%rax), %xmm2
	movaps	%xmm4, %xmm1
	movaps	(%rsi,%rax), %xmm0
	cmpltps	(%rdi,%rax), %xmm1
	cmpltps	%xmm2, %xmm0
	pand	%xmm5, %xmm0
	pand	%xmm1, %xmm0
	movaps	%xmm0, (%rcx,%rax)
	pcmpeqd	%xmm3, %xmm0
	movaps	(%rdx,%rax), %xmm1
	blendvps	%xmm0, %xmm2, %xmm1
	movaps	%xmm1, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L2
	ret
LFE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE1:
	.text
LHOTE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB3:
	.text
LHOTB3:
	.align 4,0x90
	.globl __Z4barBv
__Z4barBv:
LFB1:
	leaq	_y(%rip), %r10
	xorl	%eax, %eax
	pxor	%xmm1, %xmm1
	leaq	_w(%rip), %r9
	leaq	_x(%rip), %r8
	leaq	_b(%rip), %rdi
	.align 4,0x90
L8:
	movss	(%r10,%rax,4), %xmm0
	leaq	0(,%rax,4), %rsi
	comiss	(%r9,%rax,4), %xmm0
	seta	%cl
	comiss	(%r8,%rax,4), %xmm1
	setb	%dl
	andl	%ecx, %edx
	movb	%dl, (%rdi,%rax)
	leaq	_z(%rip), %rcx
	testb	%dl, %dl
	je	L7
	movss	(%rcx,%rax,4), %xmm0
L7:
	addq	$1, %rax
	movss	%xmm0, (%rcx,%rsi)
	cmpq	$1024, %rax
	jne	L8
	ret
LFE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE3:
	.text
LHOTE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB5:
	.text
LHOTB5:
	.align 4,0x90
	.globl __Z3barv
__Z3barv:
LFB3:
	movss	_u(%rip), %xmm8
	xorl	%eax, %eax
	pxor	%xmm4, %xmm4
	movaps	LC4(%rip), %xmm6
	leaq	_y(%rip), %rdi
	movss	_q(%rip), %xmm7
	leaq	_z(%rip), %rdx
	shufps	$0, %xmm8, %xmm8
	movdqa	LC0(%rip), %xmm5
	leaq	_x(%rip), %rsi
	leaq	_w(%rip), %rcx
	shufps	$0, %xmm7, %xmm7
	.align 4,0x90
L12:
	movaps	(%rdi,%rax), %xmm1
	movaps	%xmm6, %xmm0
	movaps	(%rsi,%rax), %xmm9
	movaps	(%rcx,%rax), %xmm2
	cmpltps	%xmm1, %xmm0
	mulps	%xmm8, %xmm9
	cmpltps	%xmm1, %xmm2
	movdqa	%xmm0, %xmm3
	pand	%xmm5, %xmm3
	movaps	%xmm2, %xmm0
	movaps	%xmm7, %xmm2
	cmpltps	%xmm9, %xmm2
	pand	%xmm3, %xmm0
	pand	%xmm2, %xmm0
	pcmpeqd	%xmm4, %xmm0
	blendvps	%xmm0, (%rdx,%rax), %xmm1
	movaps	%xmm1, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L12
	ret
LFE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE5:
	.text
LHOTE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB6:
	.text
LHOTB6:
	.align 4,0x90
	.globl __Z4bar2v
__Z4bar2v:
LFB4:
	movss	_u(%rip), %xmm8
	xorl	%eax, %eax
	pxor	%xmm4, %xmm4
	movaps	LC4(%rip), %xmm6
	leaq	_y(%rip), %rdi
	movss	_q(%rip), %xmm7
	leaq	_z(%rip), %rdx
	shufps	$0, %xmm8, %xmm8
	movdqa	LC0(%rip), %xmm5
	leaq	_x(%rip), %rsi
	leaq	_w(%rip), %rcx
	shufps	$0, %xmm7, %xmm7
	.align 4,0x90
L15:
	movaps	(%rdi,%rax), %xmm1
	movaps	%xmm6, %xmm0
	movaps	(%rsi,%rax), %xmm9
	movaps	(%rcx,%rax), %xmm2
	cmpltps	%xmm1, %xmm0
	mulps	%xmm8, %xmm9
	cmpltps	%xmm1, %xmm2
	movdqa	%xmm0, %xmm3
	pand	%xmm5, %xmm3
	movaps	%xmm2, %xmm0
	movaps	%xmm7, %xmm2
	cmpltps	%xmm9, %xmm2
	pand	%xmm3, %xmm0
	pand	%xmm2, %xmm0
	pcmpeqd	%xmm4, %xmm0
	blendvps	%xmm0, (%rdx,%rax), %xmm1
	movaps	%xmm1, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L15
	ret
LFE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE6:
	.text
LHOTE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB8:
	.text
LHOTB8:
	.align 4,0x90
	.globl __Z5barIfv
__Z5barIfv:
LFB5:
	leaq	_y(%rip), %r8
	xorl	%eax, %eax
	pxor	%xmm1, %xmm1
	leaq	_w(%rip), %rdi
	leaq	_x(%rip), %rsi
	.align 4,0x90
L19:
	movss	(%r8,%rax), %xmm0
	comiss	(%rdi,%rax), %xmm0
	seta	%cl
	comiss	(%rsi,%rax), %xmm1
	setb	%dl
	testb	%dl, %cl
	je	L18
	comiss	LC7(%rip), %xmm0
	jbe	L18
	leaq	_z(%rip), %rdx
	movss	%xmm0, (%rdx,%rax)
L18:
	addq	$4, %rax
	cmpq	$4096, %rax
	jne	L19
	ret
LFE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE8:
	.text
LHOTE8:
	.globl _q
	.zerofill __DATA,__pu_bss2,_q,4,2
	.globl _u
	.zerofill __DATA,__pu_bss2,_u,4,2
	.globl _b
	.zerofill __DATA,__pu_bss6,_b,1024,6
	.globl _k
	.zerofill __DATA,__pu_bss6,_k,4096,6
	.globl _w
	.zerofill __DATA,__pu_bss6,_w,4096,6
	.globl _z
	.zerofill __DATA,__pu_bss6,_z,4096,6
	.globl _y
	.zerofill __DATA,__pu_bss6,_y,4096,6
	.globl _x
	.zerofill __DATA,__pu_bss6,_x,4096,6
	.literal16
	.align 4
LC0:
	.long	1
	.long	1
	.long	1
	.long	1
	.align 4
LC4:
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.literal4
	.align 2
LC7:
	.long	1056964608
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
	.quad	LFB3-.
	.set L$set$6,LFE3-LFB3
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB4-.
	.set L$set$8,LFE4-LFB4
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB5-.
	.set L$set$10,LFE5-LFB5
	.quad L$set$10
	.byte	0
	.align 3
LEFDE9:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

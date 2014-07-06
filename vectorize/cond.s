	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB1:
	.text
LHOTB1:
	.align 4,0x90
	.globl __Z3barv
__Z3barv:
LFB13:
	movdqa	LC0(%rip), %xmm4
	xorl	%eax, %eax
	pxor	%xmm2, %xmm2
	pxor	%xmm3, %xmm3
	leaq	_z(%rip), %rdx
	leaq	_x(%rip), %rdi
	leaq	_w(%rip), %rsi
	leaq	_y(%rip), %rcx
	.align 4,0x90
L2:
	movaps	(%rsi,%rax), %xmm0
	movaps	%xmm2, %xmm1
	cmpltps	(%rdi,%rax), %xmm1
	cmpltps	%xmm2, %xmm0
	pand	%xmm4, %xmm0
	pand	%xmm1, %xmm0
	pcmpeqd	%xmm3, %xmm0
	movaps	(%rdx,%rax), %xmm1
	blendvps	%xmm0, (%rcx,%rax), %xmm1
	movaps	%xmm1, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L2
	ret
LFE13:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE1:
	.text
LHOTE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB3:
	.text
LHOTB3:
	.align 4,0x90
	.globl __Z4barXv
__Z4barXv:
LFB14:
	leaq	_y(%rip), %r9
	xorl	%eax, %eax
	pxor	%xmm1, %xmm1
	leaq	_w(%rip), %r8
	leaq	_x(%rip), %rdi
	leaq	_k(%rip), %rsi
	.align 4,0x90
L8:
	movss	(%r9,%rax), %xmm0
	comiss	(%r8,%rax), %xmm0
	seta	%cl
	comiss	(%rdi,%rax), %xmm1
	setb	%dl
	andl	%ecx, %edx
	movzbl	%dl, %ecx
	testb	%dl, %dl
	movl	%ecx, (%rsi,%rax)
	leaq	_z(%rip), %rcx
	je	L7
	movss	(%rcx,%rax), %xmm0
L7:
	movss	%xmm0, (%rcx,%rax)
	addq	$4, %rax
	cmpq	$4096, %rax
	jne	L8
	ret
LFE14:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE3:
	.text
LHOTE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB4:
	.text
LHOTB4:
	.align 4,0x90
	.globl __Z5barMPv
__Z5barMPv:
LFB15:
	movdqa	LC0(%rip), %xmm4
	xorl	%eax, %eax
	pxor	%xmm2, %xmm2
	pxor	%xmm3, %xmm3
	leaq	_z(%rip), %rdx
	leaq	_x(%rip), %rdi
	leaq	_w(%rip), %rsi
	leaq	_y(%rip), %rcx
	.align 4,0x90
L12:
	movaps	(%rsi,%rax), %xmm0
	movaps	%xmm2, %xmm1
	cmpltps	(%rdi,%rax), %xmm1
	cmpltps	%xmm2, %xmm0
	pand	%xmm4, %xmm0
	pand	%xmm1, %xmm0
	pcmpeqd	%xmm3, %xmm0
	movaps	(%rdx,%rax), %xmm1
	blendvps	%xmm0, (%rcx,%rax), %xmm1
	movaps	%xmm1, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L12
	ret
LFE15:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE4:
	.text
LHOTE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB5:
	.text
LHOTB5:
	.align 4,0x90
	.globl __Z6barIntv
__Z6barIntv:
LFB16:
	movdqa	LC0(%rip), %xmm4
	xorl	%eax, %eax
	pxor	%xmm2, %xmm2
	pxor	%xmm3, %xmm3
	leaq	_z(%rip), %rdx
	leaq	_x(%rip), %rdi
	leaq	_w(%rip), %rsi
	leaq	_y(%rip), %rcx
	.align 4,0x90
L15:
	movaps	%xmm2, %xmm0
	cmpleps	(%rsi,%rax), %xmm0
	movdqa	%xmm0, %xmm1
	movaps	(%rdi,%rax), %xmm0
	pand	%xmm4, %xmm1
	cmpleps	%xmm2, %xmm0
	pand	%xmm1, %xmm0
	pcmpeqd	%xmm3, %xmm0
	movaps	(%rdx,%rax), %xmm1
	blendvps	%xmm0, (%rcx,%rax), %xmm1
	movaps	%xmm1, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L15
	ret
LFE16:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE5:
	.text
LHOTE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB6:
	.text
LHOTB6:
	.align 4,0x90
	.globl __Z7barInt0v
__Z7barInt0v:
LFB17:
	movdqa	LC0(%rip), %xmm4
	xorl	%eax, %eax
	pxor	%xmm2, %xmm2
	pxor	%xmm3, %xmm3
	leaq	_z(%rip), %rdx
	leaq	_x(%rip), %rdi
	leaq	_w(%rip), %rsi
	leaq	_y(%rip), %rcx
	.align 4,0x90
L18:
	movaps	(%rsi,%rax), %xmm0
	movaps	%xmm2, %xmm1
	cmpltps	(%rdi,%rax), %xmm1
	cmpltps	%xmm2, %xmm0
	pand	%xmm4, %xmm0
	pand	%xmm1, %xmm0
	pcmpeqd	%xmm3, %xmm0
	movaps	(%rdx,%rax), %xmm1
	blendvps	%xmm0, (%rcx,%rax), %xmm1
	movaps	%xmm1, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L18
	ret
LFE17:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE6:
	.text
LHOTE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB8:
	.text
LHOTB8:
	.align 4,0x90
	.globl __Z7barPlusv
__Z7barPlusv:
LFB18:
	movdqa	LC0(%rip), %xmm3
	xorl	%eax, %eax
	pxor	%xmm2, %xmm2
	movdqa	LC7(%rip), %xmm4
	leaq	_z(%rip), %rdx
	leaq	_x(%rip), %rdi
	leaq	_w(%rip), %rsi
	leaq	_y(%rip), %rcx
	.align 4,0x90
L21:
	movaps	%xmm2, %xmm0
	cmpltps	(%rdi,%rax), %xmm0
	movdqa	%xmm0, %xmm1
	movaps	(%rsi,%rax), %xmm0
	pand	%xmm3, %xmm1
	cmpltps	%xmm2, %xmm0
	pand	%xmm3, %xmm0
	paddd	%xmm1, %xmm0
	movaps	(%rcx,%rax), %xmm1
	pcmpeqd	%xmm4, %xmm0
	blendvps	%xmm0, (%rdx,%rax), %xmm1
	movaps	%xmm1, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L21
	ret
LFE18:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE8:
	.text
LHOTE8:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB9:
	.text
LHOTB9:
	.align 4,0x90
	.globl __Z3foov
__Z3foov:
LFB19:
	leaq	_z(%rip), %rdx
	xorl	%eax, %eax
	pxor	%xmm2, %xmm2
	leaq	_j(%rip), %rdi
	leaq	_k(%rip), %rsi
	leaq	_y(%rip), %rcx
	.align 4,0x90
L24:
	movdqa	(%rdi,%rax), %xmm0
	pand	(%rsi,%rax), %xmm0
	pcmpeqd	%xmm2, %xmm0
	movaps	(%rdx,%rax), %xmm1
	blendvps	%xmm0, (%rcx,%rax), %xmm1
	movaps	%xmm1, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L24
	ret
LFE19:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE9:
	.text
LHOTE9:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB10:
	.text
LHOTB10:
	.align 4,0x90
	.globl __Z4foo2v
__Z4foo2v:
LFB20:
	movdqa	LC0(%rip), %xmm2
	xorl	%eax, %eax
	pxor	%xmm1, %xmm1
	leaq	_k(%rip), %rdi
	leaq	_x(%rip), %rsi
	leaq	_j(%rip), %rcx
	leaq	_w(%rip), %rdx
	.align 4,0x90
L27:
	movaps	%xmm1, %xmm0
	cmpltps	(%rsi,%rax), %xmm0
	pand	%xmm2, %xmm0
	movaps	%xmm0, (%rdi,%rax)
	movaps	(%rdx,%rax), %xmm0
	cmpltps	%xmm1, %xmm0
	pand	%xmm2, %xmm0
	movaps	%xmm0, (%rcx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L27
	ret
LFE20:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE10:
	.text
LHOTE10:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB11:
	.text
LHOTB11:
	.align 4,0x90
	.globl __Z4bar2v
__Z4bar2v:
LFB21:
	leaq	_x(%rip), %r10
	xorl	%eax, %eax
	pxor	%xmm0, %xmm0
	leaq	_k(%rip), %r9
	leaq	_w(%rip), %r8
	leaq	_j(%rip), %rdi
	jmp	L32
	.align 4,0x90
L35:
	movss	(%rsi,%rax), %xmm1
L31:
	movss	%xmm1, (%rsi,%rax)
	addq	$4, %rax
	cmpq	$4096, %rax
	je	L34
L32:
	comiss	(%r10,%rax), %xmm0
	setb	%dl
	comiss	(%r8,%rax), %xmm0
	movzbl	%dl, %ecx
	movl	%ecx, (%r9,%rax)
	seta	%cl
	movzbl	%cl, %esi
	testb	%dl, %cl
	movl	%esi, (%rdi,%rax)
	leaq	_z(%rip), %rsi
	jne	L35
	leaq	_y(%rip), %rdx
	movss	(%rdx,%rax), %xmm1
	jmp	L31
	.align 4,0x90
L34:
	ret
LFE21:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE11:
	.text
LHOTE11:
	.globl _j
	.zerofill __DATA,__pu_bss6,_j,4096,6
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
LC7:
	.long	2
	.long	2
	.long	2
	.long	2
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
	.quad	LFB13-.
	.set L$set$2,LFE13-LFB13
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB14-.
	.set L$set$4,LFE14-LFB14
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB15-.
	.set L$set$6,LFE15-LFB15
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB16-.
	.set L$set$8,LFE16-LFB16
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB17-.
	.set L$set$10,LFE17-LFB17
	.quad L$set$10
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$11,LEFDE11-LASFDE11
	.long L$set$11
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB18-.
	.set L$set$12,LFE18-LFB18
	.quad L$set$12
	.byte	0
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$13,LEFDE13-LASFDE13
	.long L$set$13
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB19-.
	.set L$set$14,LFE19-LFB19
	.quad L$set$14
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$15,LEFDE15-LASFDE15
	.long L$set$15
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB20-.
	.set L$set$16,LFE20-LFB20
	.quad L$set$16
	.byte	0
	.align 3
LEFDE15:
LSFDE17:
	.set L$set$17,LEFDE17-LASFDE17
	.long L$set$17
LASFDE17:
	.long	LASFDE17-EH_frame1
	.quad	LFB21-.
	.set L$set$18,LFE21-LFB21
	.quad L$set$18
	.byte	0
	.align 3
LEFDE17:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

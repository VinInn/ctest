	.text
	.align 4,0x90
	.globl __Z6myrandv
__Z6myrandv:
LFB2836:
	movq	2496+_eng(%rip), %rax
	leaq	_r(%rip), %rdi
	pxor	%xmm2, %xmm2
	leaq	2496+_r(%rip), %r9
	movdqa	LC0(%rip), %xmm6
	leaq	_eng(%rip), %r10
	movdqa	LC1(%rip), %xmm5
	leaq	896+_eng(%rip), %rsi
	movdqa	LC2(%rip), %xmm4
	movdqa	LC3(%rip), %xmm3
	leaq	2492+_eng(%rip), %rdx
	jmp	L10
	.align 4,0x90
L13:
	movl	(%r10,%rax,4), %ecx
	leaq	1(%rax), %r8
L3:
	movl	%ecx, %eax
	addq	$4, %rdi
	shrl	$11, %eax
	xorl	%ecx, %eax
	movl	%eax, %ecx
	sall	$7, %ecx
	andl	$-1658038656, %ecx
	xorl	%eax, %ecx
	movl	%ecx, %eax
	sall	$15, %eax
	andl	$-272236544, %eax
	xorl	%ecx, %eax
	movl	%eax, %ecx
	shrl	$18, %ecx
	xorl	%eax, %ecx
	movl	%ecx, -4(%rdi)
	cmpq	%r9, %rdi
	je	L9
	movq	%r8, %rax
L10:
	cmpq	$623, %rax
	jbe	L13
	leaq	_eng(%rip), %rax
	.align 4,0x90
L6:
	movdqa	(%rax), %xmm1
	addq	$16, %rax
	movdqu	-12(%rax), %xmm0
	pand	%xmm6, %xmm1
	movdqu	1572(%rax), %xmm7
	pand	%xmm5, %xmm0
	por	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	pand	%xmm4, %xmm0
	pcmpeqd	%xmm2, %xmm0
	psrld	$1, %xmm1
	pandn	%xmm3, %xmm0
	pxor	%xmm7, %xmm0
	pxor	%xmm1, %xmm0
	movdqa	%xmm0, -16(%rax)
	cmpq	%rsi, %rax
	jne	L6
	movl	900+_eng(%rip), %r8d
	movl	896+_eng(%rip), %ecx
	movl	%r8d, %eax
	andl	$-2147483648, %r8d
	andl	$2147483647, %eax
	andl	$-2147483648, %ecx
	orl	%eax, %ecx
	movl	%ecx, %eax
	shrl	%ecx
	andl	$1, %eax
	negl	%eax
	andl	$-1727483681, %eax
	xorl	2484+_eng(%rip), %eax
	xorl	%ecx, %eax
	movl	904+_eng(%rip), %ecx
	movl	%eax, 896+_eng(%rip)
	movl	%ecx, %eax
	andl	$-2147483648, %ecx
	andl	$2147483647, %eax
	orl	%eax, %r8d
	movl	%r8d, %eax
	shrl	%r8d
	andl	$1, %eax
	negl	%eax
	andl	$-1727483681, %eax
	xorl	2488+_eng(%rip), %eax
	xorl	%r8d, %eax
	movl	%eax, 900+_eng(%rip)
	movl	908+_eng(%rip), %eax
	andl	$2147483647, %eax
	orl	%eax, %ecx
	movl	%ecx, %eax
	shrl	%ecx
	andl	$1, %eax
	negl	%eax
	andl	$-1727483681, %eax
	xorl	2492+_eng(%rip), %eax
	xorl	%ecx, %eax
	movl	%eax, 904+_eng(%rip)
	leaq	908+_eng(%rip), %rax
	.align 4,0x90
L5:
	movdqu	(%rax), %xmm1
	addq	$16, %rax
	movdqa	-12(%rax), %xmm0
	pand	%xmm6, %xmm1
	pand	%xmm5, %xmm0
	por	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	pand	%xmm4, %xmm0
	pcmpeqd	%xmm2, %xmm0
	psrld	$1, %xmm1
	pandn	%xmm3, %xmm0
	pxor	-924(%rax), %xmm0
	pxor	%xmm1, %xmm0
	movdqu	%xmm0, -16(%rax)
	cmpq	%rdx, %rax
	jne	L5
	movl	_eng(%rip), %ecx
	movl	2492+_eng(%rip), %eax
	movl	%ecx, %r8d
	andl	$-2147483648, %eax
	andl	$2147483647, %r8d
	orl	%eax, %r8d
	movl	%r8d, %eax
	andl	$1, %r8d
	shrl	%eax
	negl	%r8d
	xorl	1584+_eng(%rip), %eax
	andl	$-1727483681, %r8d
	xorl	%r8d, %eax
	movl	$1, %r8d
	movl	%eax, 2492+_eng(%rip)
	jmp	L3
L9:
	movq	%r8, 2496+_eng(%rip)
	ret
LFE2836:
	.align 4,0x90
	.globl __Z5vrandv
__Z5vrandv:
LFB2837:
	leaq	_eng(%rip), %rdx
	movdqa	LC1(%rip), %xmm3
	pxor	%xmm7, %xmm7
	movdqa	LC0(%rip), %xmm2
	leaq	896+_eng(%rip), %rcx
	movq	%rdx, %rax
	movdqa	LC2(%rip), %xmm4
	movdqa	LC3(%rip), %xmm5
	.align 4,0x90
L17:
	movdqu	4(%rax), %xmm1
	addq	$16, %rax
	movdqa	-16(%rax), %xmm0
	pand	%xmm3, %xmm1
	movdqu	1572(%rax), %xmm6
	pand	%xmm2, %xmm0
	por	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	pand	%xmm4, %xmm0
	pcmpeqd	%xmm7, %xmm0
	psrld	$1, %xmm1
	pandn	%xmm5, %xmm0
	pxor	%xmm6, %xmm0
	pxor	%xmm1, %xmm0
	movdqa	%xmm0, -16(%rax)
	cmpq	%rcx, %rax
	jne	L17
	movl	900+_eng(%rip), %esi
	pxor	%xmm6, %xmm6
	movl	896+_eng(%rip), %ecx
	movl	%esi, %eax
	andl	$-2147483648, %esi
	andl	$2147483647, %eax
	andl	$-2147483648, %ecx
	orl	%eax, %ecx
	movl	%ecx, %eax
	shrl	%ecx
	andl	$1, %eax
	negl	%eax
	andl	$-1727483681, %eax
	xorl	2484+_eng(%rip), %eax
	xorl	%ecx, %eax
	movl	904+_eng(%rip), %ecx
	movl	%eax, 896+_eng(%rip)
	movl	%ecx, %eax
	andl	$-2147483648, %ecx
	andl	$2147483647, %eax
	orl	%eax, %esi
	movl	%esi, %eax
	shrl	%esi
	andl	$1, %eax
	negl	%eax
	andl	$-1727483681, %eax
	xorl	2488+_eng(%rip), %eax
	xorl	%esi, %eax
	movl	%eax, 900+_eng(%rip)
	movl	908+_eng(%rip), %eax
	andl	$2147483647, %eax
	orl	%eax, %ecx
	movl	%ecx, %eax
	shrl	%ecx
	andl	$1, %eax
	negl	%eax
	andl	$-1727483681, %eax
	xorl	2492+_eng(%rip), %eax
	xorl	%ecx, %eax
	movl	%eax, 904+_eng(%rip)
	leaq	2492+_eng(%rip), %rcx
	leaq	908+_eng(%rip), %rax
	.align 4,0x90
L16:
	movdqu	(%rax), %xmm1
	addq	$16, %rax
	movdqa	-12(%rax), %xmm0
	pand	%xmm2, %xmm1
	pand	%xmm3, %xmm0
	por	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	pand	%xmm4, %xmm0
	pcmpeqd	%xmm6, %xmm0
	psrld	$1, %xmm1
	pandn	%xmm5, %xmm0
	pxor	-924(%rax), %xmm0
	pxor	%xmm1, %xmm0
	movdqu	%xmm0, -16(%rax)
	cmpq	%rcx, %rax
	jne	L16
	movl	2492+_eng(%rip), %ecx
	movq	$0, 2496+_eng(%rip)
	movl	_eng(%rip), %eax
	movdqa	LC4(%rip), %xmm3
	movdqa	LC5(%rip), %xmm2
	andl	$-2147483648, %ecx
	andl	$2147483647, %eax
	orl	%ecx, %eax
	movl	%eax, %ecx
	andl	$1, %eax
	shrl	%ecx
	negl	%eax
	xorl	1584+_eng(%rip), %ecx
	andl	$-1727483681, %eax
	xorl	%eax, %ecx
	xorl	%eax, %eax
	movl	%ecx, 2492+_eng(%rip)
	leaq	_r(%rip), %rcx
	.align 4,0x90
L21:
	movdqa	(%rdx,%rax), %xmm1
	movdqa	%xmm1, %xmm0
	psrld	$11, %xmm0
	pxor	%xmm1, %xmm0
	movdqa	%xmm0, %xmm1
	pslld	$7, %xmm1
	pand	%xmm3, %xmm1
	pxor	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	pslld	$15, %xmm0
	pand	%xmm2, %xmm0
	pxor	%xmm1, %xmm0
	movdqa	%xmm0, %xmm1
	psrld	$18, %xmm1
	pxor	%xmm0, %xmm1
	movdqa	%xmm1, (%rcx,%rax)
	addq	$16, %rax
	cmpq	$2496, %rax
	jne	L21
	rep; ret
LFE2837:
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
__GLOBAL__sub_I_randomV.cc:
LFB3096:
	movl	$5489, %edx
	movl	$1, %ecx
	movl	$440509467, %edi
	movl	$5489, _eng(%rip)
	leaq	_eng(%rip), %r8
	.align 4
L25:
	movl	%edx, %eax
	shrl	$30, %eax
	xorl	%edx, %eax
	movl	%ecx, %edx
	shrl	$4, %edx
	imull	$1812433253, %eax, %esi
	movl	%edx, %eax
	mull	%edi
	movl	%ecx, %eax
	shrl	$2, %edx
	imull	$624, %edx, %edx
	subl	%edx, %eax
	movl	%eax, %edx
	addl	%esi, %edx
	movl	%edx, (%r8,%rcx,4)
	addq	$1, %rcx
	cmpq	$624, %rcx
	jne	L25
	movq	$624, 2496+_eng(%rip)
	ret
LFE3096:
	.globl _r
	.zerofill __DATA,__pu_bss5,_r,2496,5
	.globl _eng
	.zerofill __DATA,__pu_bss5,_eng,2504,5
	.literal16
	.align 4
LC0:
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.align 4
LC1:
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.align 4
LC2:
	.long	1
	.long	1
	.long	1
	.long	1
	.align 4
LC3:
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.align 4
LC4:
	.long	-1658038656
	.long	-1658038656
	.long	-1658038656
	.long	-1658038656
	.align 4
LC5:
	.long	-272236544
	.long	-272236544
	.long	-272236544
	.long	-272236544
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
	.quad	LFB2836-.
	.set L$set$2,LFE2836-LFB2836
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB2837-.
	.set L$set$4,LFE2837-LFB2837
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB3096-.
	.set L$set$6,LFE3096-LFB3096
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_randomV.cc
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

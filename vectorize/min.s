	.text
	.align 4,0x90
	.globl __Z4aminPKfS0_
__Z4aminPKfS0_:
LFB1641:
	leaq	4(%rdi), %rax
	vmovss	(%rdi), %xmm0
	cmpq	%rax, %rsi
	jbe	L12
	movq	%rsi, %rdx
	subq	%rdi, %rdx
	subq	$5, %rdx
	shrq	$2, %rdx
	addq	$1, %rdx
	movq	%rdx, %r8
	shrq	$3, %r8
	leaq	0(,%r8,8), %rcx
	testq	%rcx, %rcx
	je	L3
	cmpq	$7, %rdx
	jbe	L3
	vbroadcastss	%xmm0, %ymm0
	xorl	%edi, %edi
L9:
	movq	%rdi, %r9
	addq	$1, %rdi
	salq	$5, %r9
	cmpq	%rdi, %r8
	vminps	(%rax,%r9), %ymm0, %ymm0
	ja	L9
	vperm2f128	$1, %ymm0, %ymm0, %ymm1
	vminps	%ymm0, %ymm1, %ymm0
	cmpq	%rcx, %rdx
	leaq	(%rax,%rcx,4), %rax
	vshufps	$14, %ymm0, %ymm0, %ymm1
	vminps	%ymm0, %ymm1, %ymm0
	vshufps	$1, %ymm0, %ymm0, %ymm1
	vminps	%ymm0, %ymm1, %ymm0
	je	L11
	vzeroupper
L3:
	leaq	4(%rax), %rdx
	vminss	(%rax), %xmm0, %xmm0
	cmpq	%rdx, %rsi
	jbe	L12
	leaq	8(%rax), %rdx
	vminss	4(%rax), %xmm0, %xmm0
	cmpq	%rdx, %rsi
	jbe	L12
	leaq	12(%rax), %rdx
	vminss	8(%rax), %xmm0, %xmm0
	cmpq	%rdx, %rsi
	jbe	L12
	leaq	16(%rax), %rdx
	vminss	12(%rax), %xmm0, %xmm0
	cmpq	%rdx, %rsi
	jbe	L12
	leaq	20(%rax), %rdx
	vminss	16(%rax), %xmm0, %xmm0
	cmpq	%rdx, %rsi
	jbe	L12
	leaq	24(%rax), %rdx
	vminss	20(%rax), %xmm0, %xmm0
	cmpq	%rdx, %rsi
	jbe	L14
	vminss	24(%rax), %xmm0, %xmm0
	ret
	.align 4,0x90
L11:
	vzeroupper
L12:
	rep; ret
	.align 4,0x90
L14:
	rep; ret
LFE1641:
	.align 4,0x90
	.globl __Z4lminPKfS0_
__Z4lminPKfS0_:
LFB1642:
	cmpq	%rsi, %rdi
	movq	%rdi, %rax
	jae	L18
	vmovss	(%rdi), %xmm0
	movq	%rdi, %rdx
	.align 4,0x90
L17:
	vmovss	(%rdx), %xmm1
	vcomiss	%xmm1, %xmm0
	vminss	%xmm1, %xmm0, %xmm0
	cmova	%rdx, %rax
	addq	$4, %rdx
	cmpq	%rdx, %rsi
	ja	L17
L18:
	rep; ret
LFE1642:
	.align 4,0x90
	.globl __Z5lmin2PKfS0_
__Z5lmin2PKfS0_:
LFB1643:
	leaq	4(%rdi), %rdx
	movq	%rdi, %rax
	vmovss	(%rdi), %xmm0
	cmpq	%rsi, %rdx
	je	L25
	.align 4,0x90
L24:
	vmovss	(%rdx), %xmm1
	vcomiss	%xmm1, %xmm0
	vminss	%xmm1, %xmm0, %xmm0
	cmova	%rdx, %rax
	addq	$4, %rdx
	cmpq	%rdx, %rsi
	jne	L24
L25:
	rep; ret
LFE1643:
	.align 4,0x90
	.globl __Z5lmin3PKfS0_
__Z5lmin3PKfS0_:
LFB1644:
	leaq	4(%rdi), %rdx
	vmovss	(%rdi), %xmm0
	cmpq	%rsi, %rdx
	je	L33
	vxorps	%xmm3, %xmm3, %xmm3
	.align 4,0x90
L32:
	movq	%rdx, %rcx
	vmovss	(%rdx), %xmm1
	vxorps	%xmm4, %xmm4, %xmm4
	subq	%rdi, %rcx
	addq	$4, %rdx
	sarq	$2, %rcx
	vcmpltss	%xmm0, %xmm1, %xmm2
	cmpq	%rdx, %rsi
	vcvtsi2ssq	%rcx, %xmm4, %xmm4
	vminss	%xmm1, %xmm0, %xmm0
	vandnps	%xmm3, %xmm2, %xmm3
	vandps	%xmm2, %xmm4, %xmm4
	vorps	%xmm4, %xmm3, %xmm3
	jne	L32
	vaddss	LC1(%rip), %xmm3, %xmm3
	vcvttss2si	%xmm3, %eax
	cltq
	salq	$2, %rax
L27:
	addq	%rdi, %rax
	ret
L33:
	xorl	%eax, %eax
	jmp	L27
LFE1644:
	.align 4,0x90
	.globl __Z5lmin4PKfi
__Z5lmin4PKfi:
LFB1645:
	cmpl	$1, %esi
	jle	L40
	vmovss	(%rdi), %xmm0
	movl	$1, %edx
	addq	$4, %rdi
	xorl	%eax, %eax
	.align 4,0x90
L39:
	vmovss	(%rdi), %xmm1
	vcomiss	%xmm1, %xmm0
	vminss	%xmm1, %xmm0, %xmm0
	cmova	%edx, %eax
	addl	$1, %edx
	addq	$4, %rdi
	cmpl	%esi, %edx
	jne	L39
	rep; ret
L40:
	xorl	%eax, %eax
	ret
LFE1645:
	.align 4,0x90
	.globl __Z5lmin5PKfi
__Z5lmin5PKfi:
LFB1646:
	cmpl	$1, %esi
	vmovss	(%rdi), %xmm0
	je	L48
	addq	$4, %rdi
	movl	$1, %edx
	xorl	%eax, %eax
	.align 4,0x90
L47:
	vmovss	(%rdi), %xmm1
	vcomiss	%xmm1, %xmm0
	vminss	%xmm1, %xmm0, %xmm0
	cmova	%edx, %eax
	addl	$1, %edx
	addq	$4, %rdi
	cmpl	%esi, %edx
	jne	L47
	rep; ret
L48:
	xorl	%eax, %eax
	ret
LFE1646:
	.align 4,0x90
	.globl __Z5lmin6PKfPii
__Z5lmin6PKfPii:
LFB1647:
	cmpl	$1, %edx
	movl	$0, (%rsi)
	vmovss	(%rdi), %xmm0
	je	L56
	subl	$2, %edx
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	leaq	4(,%rdx,4), %r8
	.align 4,0x90
L52:
	vmovss	4(%rdi,%rax), %xmm1
	movl	4(%rsi,%rax), %edx
	vcomiss	%xmm0, %xmm1
	vminss	%xmm1, %xmm0, %xmm0
	cmovb	%ecx, %edx
	movl	%edx, 4(%rsi,%rax)
	addq	$4, %rax
	cmpq	%r8, %rax
	jne	L52
L56:
	rep; ret
LFE1647:
	.align 4,0x90
	.globl __Z4fminPKfi
__Z4fminPKfi:
LFB1648:
	cmpl	$1, %esi
	vmovss	(%rdi), %xmm0
	je	L90
	leaq	4(%rdi), %rax
	leal	-1(%rsi), %ecx
	andl	$31, %eax
	shrq	$2, %rax
	negq	%rax
	andl	$7, %eax
	cmpl	%ecx, %eax
	cmova	%ecx, %eax
	cmpl	$10, %ecx
	movl	%eax, %edx
	movl	%ecx, %eax
	ja	L91
L70:
	cmpl	$1, %eax
	vminss	4(%rdi), %xmm0, %xmm0
	je	L72
	cmpl	$2, %eax
	vminss	8(%rdi), %xmm0, %xmm0
	je	L73
	cmpl	$3, %eax
	vminss	12(%rdi), %xmm0, %xmm0
	je	L74
	cmpl	$4, %eax
	vminss	16(%rdi), %xmm0, %xmm0
	je	L75
	cmpl	$5, %eax
	vminss	20(%rdi), %xmm0, %xmm0
	je	L76
	cmpl	$6, %eax
	vminss	24(%rdi), %xmm0, %xmm0
	je	L77
	cmpl	$7, %eax
	vminss	28(%rdi), %xmm0, %xmm0
	je	L78
	cmpl	$8, %eax
	vminss	32(%rdi), %xmm0, %xmm0
	je	L79
	cmpl	$10, %eax
	vminss	36(%rdi), %xmm0, %xmm0
	jne	L80
	vminss	40(%rdi), %xmm0, %xmm0
	movl	$11, %edx
L61:
	cmpl	%eax, %ecx
	je	L92
L60:
	subl	%eax, %ecx
	movl	%eax, %r8d
	movl	%ecx, %r10d
	shrl	$3, %r10d
	leal	0(,%r10,8), %r9d
	testl	%r9d, %r9d
	je	L63
	leaq	4(%rdi,%r8,4), %r8
	vbroadcastss	%xmm0, %ymm0
	xorl	%eax, %eax
L69:
	addl	$1, %eax
	vminps	(%r8), %ymm0, %ymm0
	addq	$32, %r8
	cmpl	%eax, %r10d
	ja	L69
	vperm2f128	$1, %ymm0, %ymm0, %ymm1
	vminps	%ymm0, %ymm1, %ymm0
	addl	%r9d, %edx
	cmpl	%r9d, %ecx
	vshufps	$14, %ymm0, %ymm0, %ymm1
	vminps	%ymm0, %ymm1, %ymm0
	vshufps	$1, %ymm0, %ymm0, %ymm1
	vminps	%ymm0, %ymm1, %ymm0
	je	L89
	vzeroupper
L63:
	movslq	%edx, %rax
	vminss	(%rdi,%rax,4), %xmm0, %xmm0
	leal	1(%rdx), %eax
	cmpl	%eax, %esi
	je	L90
	cltq
	vminss	(%rdi,%rax,4), %xmm0, %xmm0
	leal	2(%rdx), %eax
	cmpl	%eax, %esi
	je	L90
	cltq
	vminss	(%rdi,%rax,4), %xmm0, %xmm0
	leal	3(%rdx), %eax
	cmpl	%eax, %esi
	je	L90
	cltq
	vminss	(%rdi,%rax,4), %xmm0, %xmm0
	leal	4(%rdx), %eax
	cmpl	%eax, %esi
	je	L90
	cltq
	vminss	(%rdi,%rax,4), %xmm0, %xmm0
	leal	5(%rdx), %eax
	cmpl	%eax, %esi
	je	L90
	addl	$6, %edx
	cltq
	cmpl	%edx, %esi
	vminss	(%rdi,%rax,4), %xmm0, %xmm0
	je	L93
	movslq	%edx, %rdx
	vminss	(%rdi,%rdx,4), %xmm0, %xmm0
	ret
	.align 4,0x90
L89:
	vzeroupper
L90:
	rep; ret
	.align 4,0x90
L92:
	rep; ret
	.align 4,0x90
L91:
	testl	%edx, %edx
	movl	%edx, %eax
	jne	L70
	movl	$1, %edx
	.p2align 4,,2
	jmp	L60
	.align 4,0x90
L93:
	rep; ret
	.align 4,0x90
L80:
	movl	$10, %edx
	jmp	L61
	.align 4,0x90
L74:
	movl	$4, %edx
	jmp	L61
	.align 4,0x90
L75:
	movl	$5, %edx
	jmp	L61
	.align 4,0x90
L72:
	movl	$2, %edx
	jmp	L61
	.align 4,0x90
L73:
	movl	$3, %edx
	jmp	L61
	.align 4,0x90
L76:
	movl	$6, %edx
	jmp	L61
	.align 4,0x90
L77:
	movl	$7, %edx
	jmp	L61
	.align 4,0x90
L78:
	movl	$8, %edx
	jmp	L61
	.align 4,0x90
L79:
	movl	$9, %edx
	jmp	L61
LFE1648:
	.literal4
	.align 2
LC1:
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
	.quad	LFB1641-.
	.set L$set$2,LFE1641-LFB1641
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1642-.
	.set L$set$4,LFE1642-LFB1642
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB1643-.
	.set L$set$6,LFE1643-LFB1643
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB1644-.
	.set L$set$8,LFE1644-LFB1644
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB1645-.
	.set L$set$10,LFE1645-LFB1645
	.quad L$set$10
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$11,LEFDE11-LASFDE11
	.long L$set$11
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB1646-.
	.set L$set$12,LFE1646-LFB1646
	.quad L$set$12
	.byte	0
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$13,LEFDE13-LASFDE13
	.long L$set$13
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB1647-.
	.set L$set$14,LFE1647-LFB1647
	.quad L$set$14
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$15,LEFDE15-LASFDE15
	.long L$set$15
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB1648-.
	.set L$set$16,LFE1648-LFB1648
	.quad L$set$16
	.byte	0
	.align 3
LEFDE15:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

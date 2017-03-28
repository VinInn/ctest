	.text
	.align 4,0x90
	.globl __Z3f20v
__Z3f20v:
LFB0:
	vmovaps	LC0(%rip), %ymm2
	xorl	%eax, %eax
	leaq	_k(%rip), %r8
	leaq	_fx(%rip), %rdi
	leaq	_fy(%rip), %rsi
	leaq	_fz(%rip), %rcx
	leaq	_g(%rip), %rdx
	.align 4,0x90
L2:
	vmovdqa	(%r8,%rax), %ymm1
	vmovaps	%ymm2, %ymm5
	vmovaps	%ymm2, %ymm6
	vmovaps	%ymm2, %ymm7
	vgatherdps	%ymm5, (%rdi,%ymm1,4), %ymm4
	vgatherdps	%ymm6, (%rsi,%ymm1,4), %ymm0
	vgatherdps	%ymm7, (%rcx,%ymm1,4), %ymm3
	vaddps	%ymm4, %ymm0, %ymm0
	vaddps	%ymm3, %ymm0, %ymm0
	vmovaps	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L2
	vzeroupper
	ret
LFE0:
	.align 4,0x90
	.globl __Z3f21v
__Z3f21v:
LFB1:
	vmovdqa	LC1(%rip), %ymm7
	xorl	%eax, %eax
	vmovaps	LC0(%rip), %ymm2
	leaq	_k(%rip), %rsi
	leaq	_ff(%rip), %rdx
	leaq	_g(%rip), %rcx
	vmovdqa	LC2(%rip), %ymm6
	vmovdqa	LC3(%rip), %ymm5
	.align 4,0x90
L6:
	vmovaps	%ymm2, %ymm3
	vpmulld	(%rsi,%rax), %ymm7, %ymm1
	vmovaps	%ymm2, %ymm8
	vmovaps	%ymm2, %ymm9
	vgatherdps	%ymm3, (%rdx,%ymm1,4), %ymm4
	vpaddd	%ymm6, %ymm1, %ymm3
	vpaddd	%ymm5, %ymm1, %ymm1
	vgatherdps	%ymm8, (%rdx,%ymm3,4), %ymm0
	vgatherdps	%ymm9, (%rdx,%ymm1,4), %ymm3
	vaddps	%ymm4, %ymm0, %ymm0
	vaddps	%ymm3, %ymm0, %ymm0
	vmovaps	%ymm0, (%rcx,%rax)
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L6
	vzeroupper
	ret
LFE1:
	.align 4,0x90
	.globl __Z4f21bv
__Z4f21bv:
LFB2:
	xorl	%eax, %eax
	leaq	_k(%rip), %rdi
	leaq	_ff(%rip), %rsi
	leaq	_g(%rip), %rcx
	.align 4,0x90
L9:
	movl	(%rdi,%rax), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	leaq	(%rsi,%rdx,4), %rdx
	vmovss	(%rdx), %xmm0
	vaddss	4(%rdx), %xmm0, %xmm0
	vaddss	8(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, (%rcx,%rax)
	addq	$4, %rax
	cmpq	$4096, %rax
	jne	L9
	ret
LFE2:
	.align 4,0x90
	.globl __Z3f22v
__Z3f22v:
LFB3:
	xorl	%eax, %eax
	leaq	_k(%rip), %rdi
	leaq	_g(%rip), %rsi
	leaq	_f3(%rip), %rcx
	.align 4,0x90
L12:
	movslq	(%rdi,%rax), %rdx
	leaq	(%rdx,%rdx,2), %rdx
	leaq	(%rcx,%rdx,4), %rdx
	vmovss	(%rdx), %xmm0
	vaddss	4(%rdx), %xmm0, %xmm0
	vaddss	8(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi,%rax)
	addq	$4, %rax
	cmpq	$4096, %rax
	jne	L12
	ret
LFE3:
	.align 4,0x90
	.globl __Z3bar6float3iii
__Z3bar6float3iii:
LFB4:
	vmovq	%xmm0, -16(%rsp)
	vmovss	-12(%rsp), %xmm3
	testl	%edi, %edi
	jle	L18
	leal	-1(%rdi), %ecx
	vmovss	-16(%rsp), %xmm4
	xorl	%eax, %eax
	leaq	_neighList(%rip), %r8
	addq	$1, %rcx
	leaq	_position(%rip), %rdi
	leaq	_r2inv(%rip), %rsi
	.align 4,0x90
L16:
	movslq	(%r8,%rax,4), %rdx
	leaq	(%rdx,%rdx,2), %rdx
	leaq	(%rdi,%rdx,4), %rdx
	vsubss	4(%rdx), %xmm3, %xmm2
	vsubss	(%rdx), %xmm4, %xmm1
	vsubss	8(%rdx), %xmm3, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, (%rsi,%rax,4)
	addq	$1, %rax
	cmpq	%rax, %rcx
	jne	L16
L18:
	ret
LFE4:
	.align 4,0x90
	.globl __Z4bar26float3iii
__Z4bar26float3iii:
LFB5:
	leaq	8(%rsp), %r10
LCFI0:
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
LCFI1:
	movq	%rsp, %rbp
	pushq	%r10
LCFI2:
	vmovq	%xmm0, -32(%rbp)
	vmovss	%xmm1, -24(%rbp)
	testl	%edi, %edi
	jle	L27
	leal	-1(%rdi), %eax
	vmovss	-32(%rbp), %xmm0
	vmovss	-28(%rbp), %xmm2
	cmpl	$6, %eax
	jbe	L24
	movl	%edi, %r11d
	vbroadcastss	%xmm0, %ymm11
	vbroadcastss	%xmm2, %ymm10
	xorl	%edx, %edx
	vmovdqa	LC1(%rip), %ymm8
	shrl	$3, %r11d
	xorl	%ecx, %ecx
	vmovaps	LC0(%rip), %ymm7
	leaq	4+_position(%rip), %r8
	leaq	8+_position(%rip), %r10
	vbroadcastss	%xmm1, %ymm9
	leaq	_neighList(%rip), %r9
	leaq	_position(%rip), %rax
	leaq	_r2inv(%rip), %rsi
L22:
	vmovaps	%ymm7, %ymm6
	vmovaps	%ymm7, %ymm4
	vmovaps	%ymm7, %ymm12
	addl	$1, %ecx
	vpmulld	(%r9,%rdx), %ymm8, %ymm3
	vgatherdps	%ymm6, (%rax,%ymm3,4), %ymm5
	vgatherdps	%ymm4, (%r8,%ymm3,4), %ymm6
	vgatherdps	%ymm12, (%r10,%ymm3,4), %ymm4
	vsubps	%ymm6, %ymm10, %ymm6
	vsubps	%ymm5, %ymm11, %ymm5
	vsubps	%ymm4, %ymm9, %ymm3
	vmulps	%ymm6, %ymm6, %ymm6
	vfmadd132ps	%ymm5, %ymm6, %ymm5
	vfmadd132ps	%ymm3, %ymm5, %ymm3
	vmovaps	%ymm3, (%rsi,%rdx)
	addq	$32, %rdx
	cmpl	%ecx, %r11d
	ja	L22
	movl	%edi, %edx
	andl	$-8, %edx
	cmpl	%edi, %edx
	je	L29
	vzeroupper
L21:
	movslq	%edx, %r10
	movl	(%r9,%r10,4), %ecx
	leal	(%rcx,%rcx,2), %ecx
	movslq	%ecx, %rcx
	vsubss	(%rax,%rcx,4), %xmm0, %xmm3
	addq	$1, %rcx
	vsubss	(%r8,%rcx,4), %xmm1, %xmm5
	vsubss	(%rax,%rcx,4), %xmm2, %xmm4
	leal	1(%rdx), %ecx
	vmulss	%xmm5, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vfmadd132ss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm3, (%rsi,%r10,4)
	cmpl	%ecx, %edi
	jle	L27
	movslq	%ecx, %r10
	movl	(%r9,%r10,4), %ecx
	leal	(%rcx,%rcx,2), %ecx
	movslq	%ecx, %rcx
	vsubss	(%rax,%rcx,4), %xmm0, %xmm3
	addq	$1, %rcx
	vsubss	(%r8,%rcx,4), %xmm1, %xmm5
	vsubss	(%rax,%rcx,4), %xmm2, %xmm4
	leal	2(%rdx), %ecx
	vmulss	%xmm5, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vfmadd132ss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm3, (%rsi,%r10,4)
	cmpl	%ecx, %edi
	jle	L27
	movslq	%ecx, %r10
	movl	(%r9,%r10,4), %ecx
	leal	(%rcx,%rcx,2), %ecx
	movslq	%ecx, %rcx
	vsubss	(%rax,%rcx,4), %xmm0, %xmm3
	addq	$1, %rcx
	vsubss	(%r8,%rcx,4), %xmm1, %xmm5
	vsubss	(%rax,%rcx,4), %xmm2, %xmm4
	leal	3(%rdx), %ecx
	vmulss	%xmm5, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vfmadd132ss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm3, (%rsi,%r10,4)
	cmpl	%ecx, %edi
	jle	L27
	movslq	%ecx, %r10
	movl	(%r9,%r10,4), %ecx
	leal	(%rcx,%rcx,2), %ecx
	movslq	%ecx, %rcx
	vsubss	(%rax,%rcx,4), %xmm0, %xmm3
	addq	$1, %rcx
	vsubss	(%r8,%rcx,4), %xmm1, %xmm5
	vsubss	(%rax,%rcx,4), %xmm2, %xmm4
	leal	4(%rdx), %ecx
	vmulss	%xmm5, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vfmadd132ss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm3, (%rsi,%r10,4)
	cmpl	%ecx, %edi
	jle	L27
	movslq	%ecx, %r10
	movl	(%r9,%r10,4), %ecx
	leal	(%rcx,%rcx,2), %ecx
	movslq	%ecx, %rcx
	vsubss	(%rax,%rcx,4), %xmm0, %xmm3
	addq	$1, %rcx
	vsubss	(%r8,%rcx,4), %xmm1, %xmm5
	vsubss	(%rax,%rcx,4), %xmm2, %xmm4
	leal	5(%rdx), %ecx
	vmulss	%xmm5, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vfmadd132ss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm3, (%rsi,%r10,4)
	cmpl	%ecx, %edi
	jle	L27
	movslq	%ecx, %r10
	addl	$6, %edx
	movl	(%r9,%r10,4), %ecx
	leal	(%rcx,%rcx,2), %ecx
	movslq	%ecx, %rcx
	vsubss	(%rax,%rcx,4), %xmm0, %xmm3
	addq	$1, %rcx
	vsubss	(%r8,%rcx,4), %xmm1, %xmm5
	vsubss	(%rax,%rcx,4), %xmm2, %xmm4
	vmulss	%xmm5, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vfmadd132ss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm3, (%rsi,%r10,4)
	cmpl	%edx, %edi
	jle	L27
	movslq	%edx, %rdx
	movl	(%r9,%rdx,4), %ecx
	leal	(%rcx,%rcx,2), %ecx
	movslq	%ecx, %rcx
	vsubss	(%rax,%rcx,4), %xmm0, %xmm0
	addq	$1, %rcx
	vsubss	(%r8,%rcx,4), %xmm1, %xmm1
	vsubss	(%rax,%rcx,4), %xmm2, %xmm2
	vmulss	%xmm1, %xmm1, %xmm1
	vfmadd132ss	%xmm2, %xmm1, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, (%rsi,%rdx,4)
L27:
	popq	%r10
LCFI3:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI4:
	ret
	.align 4,0x90
L29:
LCFI5:
	vzeroupper
	popq	%r10
LCFI6:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI7:
	ret
	.align 4,0x90
L24:
LCFI8:
	xorl	%edx, %edx
	leaq	4+_position(%rip), %r8
	leaq	_neighList(%rip), %r9
	leaq	_position(%rip), %rax
	leaq	_r2inv(%rip), %rsi
	jmp	L21
LFE5:
	.align 4,0x90
	.globl __Z3foo6float3iii
__Z3foo6float3iii:
LFB6:
	vmovq	%xmm0, -16(%rsp)
	testl	%edi, %edi
	jle	L34
	leal	-1(%rdi), %ecx
	vmovss	-16(%rsp), %xmm5
	vmovss	-12(%rsp), %xmm4
	xorl	%eax, %eax
	addq	$1, %rcx
	leaq	_neighList(%rip), %r8
	leaq	_position(%rip), %rdi
	leaq	_r2inv(%rip), %rsi
	.align 4,0x90
L32:
	movslq	(%r8,%rax,4), %rdx
	leaq	(%rdx,%rdx,2), %rdx
	leaq	(%rdi,%rdx,4), %rdx
	vsubss	4(%rdx), %xmm4, %xmm3
	vsubss	(%rdx), %xmm5, %xmm2
	vsubss	8(%rdx), %xmm1, %xmm0
	vmulss	%xmm3, %xmm3, %xmm3
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, (%rsi,%rax,4)
	addq	$1, %rax
	cmpq	%rax, %rcx
	jne	L32
L34:
	ret
LFE6:
	.globl _r2inv
	.zerofill __DATA,__pu_bss5,_r2inv,4096,5
	.globl _maxNeighbors
	.zerofill __DATA,__pu_bss2,_maxNeighbors,4,2
	.globl _neighList
	.zerofill __DATA,__pu_bss5,_neighList,4096,5
	.globl _position
	.zerofill __DATA,__pu_bss5,_position,12288,5
	.globl _f3
	.zerofill __DATA,__pu_bss5,_f3,12288,5
	.globl _ff
	.zerofill __DATA,__pu_bss5,_ff,12288,5
	.globl _k
	.zerofill __DATA,__pu_bss5,_k,4096,5
	.globl _fz
	.zerofill __DATA,__pu_bss5,_fz,4096,5
	.globl _fy
	.zerofill __DATA,__pu_bss5,_fy,4096,5
	.globl _g
	.zerofill __DATA,__pu_bss5,_g,4096,5
	.globl _fx
	.zerofill __DATA,__pu_bss5,_fx,4096,5
	.const
	.align 5
LC0:
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.align 5
LC1:
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
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
	.long	2
	.long	2
	.long	2
	.long	2
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
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB4-.
	.set L$set$10,LFE4-LFB4
	.quad L$set$10
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$11,LEFDE11-LASFDE11
	.long L$set$11
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB5-.
	.set L$set$12,LFE5-LFB5
	.quad L$set$12
	.byte	0
	.byte	0x4
	.set L$set$13,LCFI0-LFB5
	.long L$set$13
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$14,LCFI1-LCFI0
	.long L$set$14
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$15,LCFI2-LCFI1
	.long L$set$15
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$16,LCFI3-LCFI2
	.long L$set$16
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$17,LCFI4-LCFI3
	.long L$set$17
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$18,LCFI5-LCFI4
	.long L$set$18
	.byte	0xb
	.byte	0x4
	.set L$set$19,LCFI6-LCFI5
	.long L$set$19
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$20,LCFI7-LCFI6
	.long L$set$20
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$21,LCFI8-LCFI7
	.long L$set$21
	.byte	0xb
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$22,LEFDE13-LASFDE13
	.long L$set$22
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB6-.
	.set L$set$23,LFE6-LFB6
	.quad L$set$23
	.byte	0
	.align 3
LEFDE13:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

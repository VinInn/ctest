	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB1:
	.text
LHOTB1:
	.align 4,0x90
	.globl __Z3f20v
__Z3f20v:
LFB0:
	vmovaps	LC0(%rip), %ymm3
	leaq	8(%rsp), %r10
LCFI0:
	andq	$-32, %rsp
	xorl	%eax, %eax
	pushq	-8(%r10)
	pushq	%rbp
	leaq	_k(%rip), %r8
	leaq	_fx(%rip), %rdi
LCFI1:
	movq	%rsp, %rbp
	leaq	_fy(%rip), %rsi
	pushq	%r10
LCFI2:
	leaq	_fz(%rip), %rcx
	leaq	_g(%rip), %rdx
	.align 4,0x90
L2:
	vmovdqa	(%r8,%rax), %ymm2
	vmovaps	%ymm3, %ymm6
	vmovaps	%ymm3, %ymm7
	vmovaps	%ymm3, %ymm5
	vgatherdps	%ymm6, (%rsi,%ymm2,4), %ymm1
	vgatherdps	%ymm7, (%rcx,%ymm2,4), %ymm0
	vgatherdps	%ymm5, (%rdi,%ymm2,4), %ymm4
	vaddps	%ymm0, %ymm1, %ymm0
	vaddps	%ymm4, %ymm0, %ymm0
	vmovaps	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L2
	vzeroupper
	popq	%r10
LCFI3:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI4:
	ret
LFE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE1:
	.text
LHOTE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB5:
	.text
LHOTB5:
	.align 4,0x90
	.globl __Z3f21v
__Z3f21v:
LFB1:
	vmovdqa	LC2(%rip), %ymm7
	leaq	8(%rsp), %r10
LCFI5:
	xorl	%eax, %eax
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	leaq	_k(%rip), %rsi
	vmovaps	LC0(%rip), %ymm3
	leaq	_ff(%rip), %rdx
LCFI6:
	movq	%rsp, %rbp
	vmovdqa	LC3(%rip), %ymm6
	pushq	%r10
LCFI7:
	leaq	_g(%rip), %rcx
	vmovdqa	LC4(%rip), %ymm5
	.align 4,0x90
L7:
	vmovaps	%ymm3, %ymm1
	vpmulld	(%rsi,%rax), %ymm7, %ymm2
	vmovaps	%ymm3, %ymm8
	vpaddd	%ymm6, %ymm2, %ymm0
	vgatherdps	%ymm1, (%rdx,%ymm2,4), %ymm4
	vmovaps	%ymm3, %ymm9
	vpaddd	%ymm5, %ymm2, %ymm2
	vgatherdps	%ymm8, (%rdx,%ymm0,4), %ymm1
	vgatherdps	%ymm9, (%rdx,%ymm2,4), %ymm0
	vaddps	%ymm0, %ymm1, %ymm0
	vaddps	%ymm4, %ymm0, %ymm0
	vmovaps	%ymm0, (%rcx,%rax)
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L7
	vzeroupper
	popq	%r10
LCFI8:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI9:
	ret
LFE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE5:
	.text
LHOTE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB6:
	.text
LHOTB6:
	.align 4,0x90
	.globl __Z4f21bv
__Z4f21bv:
LFB2:
	leaq	_k(%rip), %rdi
	xorl	%eax, %eax
	leaq	_ff(%rip), %rsi
	leaq	_g(%rip), %rcx
	.align 4,0x90
L11:
	movl	(%rdi,%rax), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	leaq	(%rsi,%rdx,4), %rdx
	vmovss	4(%rdx), %xmm0
	vaddss	(%rdx), %xmm0, %xmm0
	vaddss	8(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, (%rcx,%rax)
	addq	$4, %rax
	cmpq	$4096, %rax
	jne	L11
	ret
LFE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE6:
	.text
LHOTE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB7:
	.text
LHOTB7:
	.align 4,0x90
	.globl __Z3f22v
__Z3f22v:
LFB3:
	leaq	_k(%rip), %rdi
	xorl	%eax, %eax
	leaq	_g(%rip), %rsi
	leaq	_f3(%rip), %rcx
	.align 4,0x90
L14:
	movslq	(%rdi,%rax), %rdx
	leaq	(%rdx,%rdx,2), %rdx
	leaq	(%rcx,%rdx,4), %rdx
	vmovss	4(%rdx), %xmm0
	vaddss	(%rdx), %xmm0, %xmm0
	vaddss	8(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi,%rax)
	addq	$4, %rax
	cmpq	$4096, %rax
	jne	L14
	ret
LFE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE7:
	.text
LHOTE7:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB8:
	.text
LHOTB8:
	.align 4,0x90
	.globl __Z3bar6float3iii
__Z3bar6float3iii:
LFB4:
	vmovq	%xmm0, -16(%rsp)
	vmovss	-12(%rsp), %xmm3
	testl	%edi, %edi
	jle	L19
	subl	$1, %edi
	vmovss	-16(%rsp), %xmm4
	xorl	%eax, %eax
	leaq	4(,%rdi,4), %r8
	leaq	_position(%rip), %rsi
	leaq	_neighList(%rip), %rdi
	leaq	_r2inv(%rip), %rcx
	.align 4,0x90
L18:
	movslq	(%rdi,%rax), %rdx
	leaq	(%rdx,%rdx,2), %rdx
	leaq	(%rsi,%rdx,4), %rdx
	vsubss	4(%rdx), %xmm3, %xmm2
	vsubss	(%rdx), %xmm4, %xmm1
	vsubss	8(%rdx), %xmm3, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, (%rcx,%rax)
	addq	$4, %rax
	cmpq	%r8, %rax
	jne	L18
L19:
	ret
LFE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE8:
	.text
LHOTE8:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB9:
	.text
LHOTB9:
	.align 4,0x90
	.globl __Z4bar26float3iii
__Z4bar26float3iii:
LFB5:
	leaq	8(%rsp), %r10
LCFI10:
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
LCFI11:
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
LCFI12:
	pushq	%rbx
LCFI13:
	vmovq	%xmm0, -64(%rbp)
	vmovss	%xmm1, -56(%rbp)
	testl	%edi, %edi
	jle	L31
	imull	%edi, %esi
	leal	-8(%rdi), %eax
	vmovss	-64(%rbp), %xmm4
	imull	_maxNeighbors(%rip), %edx
	leal	-1(%rdi), %ecx
	shrl	$3, %eax
	addl	$1, %eax
	vmovss	-60(%rbp), %xmm5
	movl	%esi, %r11d
	movl	%edx, %r10d
	leal	0(,%rax,8), %edx
	cmpl	$6, %ecx
	jbe	L28
	movslq	%esi, %rcx
	movslq	%r10d, %rsi
	xorl	%ebx, %ebx
	vmovdqa	LC2(%rip), %ymm7
	addq	%rsi, %rcx
	vbroadcastss	%xmm4, %ymm10
	vbroadcastss	%xmm5, %ymm9
	vmovaps	LC0(%rip), %ymm6
	leaq	_neighList(%rip), %r8
	vbroadcastss	%xmm1, %ymm8
	xorl	%r12d, %r12d
	leaq	4+_position(%rip), %rsi
	leaq	(%r8,%rcx,4), %r15
	movq	%rsi, %r14
	leaq	8+_position(%rip), %r13
	leaq	_position(%rip), %rcx
	leaq	_r2inv(%rip), %r9
L27:
	vmovaps	%ymm6, %ymm2
	vmovaps	%ymm6, %ymm3
	vmovaps	%ymm6, %ymm12
	vpmulld	(%r15,%rbx), %ymm7, %ymm11
	vgatherdps	%ymm2, (%rcx,%ymm11,4), %ymm0
	addl	$1, %r12d
	vgatherdps	%ymm3, (%r14,%ymm11,4), %ymm2
	vgatherdps	%ymm12, 0(%r13,%ymm11,4), %ymm3
	vsubps	%ymm0, %ymm10, %ymm0
	vsubps	%ymm2, %ymm9, %ymm2
	vsubps	%ymm3, %ymm8, %ymm3
	vmulps	%ymm3, %ymm3, %ymm3
	vfmadd132ps	%ymm2, %ymm3, %ymm2
	vfmadd132ps	%ymm0, %ymm2, %ymm0
	vmovaps	%ymm0, (%r9,%rbx)
	addq	$32, %rbx
	cmpl	%r12d, %eax
	ja	L27
	cmpl	%edx, %edi
	je	L34
	vzeroupper
L22:
	addl	%r11d, %r10d
	.align 4,0x90
L26:
	leal	(%r10,%rdx), %eax
	cltq
	movl	(%r8,%rax,4), %eax
	leal	(%rax,%rax,2), %eax
	cltq
	vsubss	(%rcx,%rax,4), %xmm4, %xmm0
	addq	$1, %rax
	vsubss	(%rsi,%rax,4), %xmm1, %xmm3
	vsubss	(%rcx,%rax,4), %xmm5, %xmm2
	movslq	%edx, %rax
	addl	$1, %edx
	vmulss	%xmm3, %xmm3, %xmm3
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, (%r9,%rax,4)
	cmpl	%edx, %edi
	jg	L26
L31:
	popq	%rbx
	popq	%r10
LCFI14:
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI15:
	ret
	.align 4,0x90
L34:
LCFI16:
	vzeroupper
	jmp	L31
	.align 4,0x90
L28:
	leaq	_neighList(%rip), %r8
	xorl	%edx, %edx
	leaq	4+_position(%rip), %rsi
	leaq	_position(%rip), %rcx
	leaq	_r2inv(%rip), %r9
	jmp	L22
LFE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE9:
	.text
LHOTE9:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB10:
	.text
LHOTB10:
	.align 4,0x90
	.globl __Z3foo6float3iii
__Z3foo6float3iii:
LFB6:
	vmovq	%xmm0, -16(%rsp)
	testl	%edi, %edi
	jle	L38
	subl	$1, %edi
	vmovss	-16(%rsp), %xmm5
	xorl	%eax, %eax
	vmovss	-12(%rsp), %xmm4
	leaq	4(,%rdi,4), %r8
	leaq	_position(%rip), %rsi
	leaq	_neighList(%rip), %rdi
	leaq	_r2inv(%rip), %rcx
	.align 4,0x90
L37:
	movslq	(%rdi,%rax), %rdx
	leaq	(%rdx,%rdx,2), %rdx
	leaq	(%rsi,%rdx,4), %rdx
	vsubss	4(%rdx), %xmm4, %xmm3
	vsubss	(%rdx), %xmm5, %xmm2
	vsubss	8(%rdx), %xmm1, %xmm0
	vmulss	%xmm3, %xmm3, %xmm3
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, (%rcx,%rax)
	addq	$4, %rax
	cmpq	%r8, %rax
	jne	L37
L38:
	ret
LFE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE10:
	.text
LHOTE10:
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
LC2:
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.align 5
LC3:
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.align 5
LC4:
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
	.byte	0x4
	.set L$set$3,LCFI0-LFB0
	.long L$set$3
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$6,LCFI3-LCFI2
	.long L$set$6
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$7,LCFI4-LCFI3
	.long L$set$7
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$8,LEFDE3-LASFDE3
	.long L$set$8
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1-.
	.set L$set$9,LFE1-LFB1
	.quad L$set$9
	.byte	0
	.byte	0x4
	.set L$set$10,LCFI5-LFB1
	.long L$set$10
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$11,LCFI6-LCFI5
	.long L$set$11
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$12,LCFI7-LCFI6
	.long L$set$12
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$13,LCFI8-LCFI7
	.long L$set$13
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$14,LCFI9-LCFI8
	.long L$set$14
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$15,LEFDE5-LASFDE5
	.long L$set$15
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB2-.
	.set L$set$16,LFE2-LFB2
	.quad L$set$16
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$17,LEFDE7-LASFDE7
	.long L$set$17
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB3-.
	.set L$set$18,LFE3-LFB3
	.quad L$set$18
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$19,LEFDE9-LASFDE9
	.long L$set$19
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB4-.
	.set L$set$20,LFE4-LFB4
	.quad L$set$20
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$21,LEFDE11-LASFDE11
	.long L$set$21
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB5-.
	.set L$set$22,LFE5-LFB5
	.quad L$set$22
	.byte	0
	.byte	0x4
	.set L$set$23,LCFI10-LFB5
	.long L$set$23
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$24,LCFI11-LCFI10
	.long L$set$24
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$25,LCFI12-LCFI11
	.long L$set$25
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x58
	.byte	0x6
	.byte	0x10
	.byte	0xf
	.byte	0x2
	.byte	0x76
	.byte	0x78
	.byte	0x10
	.byte	0xe
	.byte	0x2
	.byte	0x76
	.byte	0x70
	.byte	0x10
	.byte	0xd
	.byte	0x2
	.byte	0x76
	.byte	0x68
	.byte	0x10
	.byte	0xc
	.byte	0x2
	.byte	0x76
	.byte	0x60
	.byte	0x4
	.set L$set$26,LCFI13-LCFI12
	.long L$set$26
	.byte	0x10
	.byte	0x3
	.byte	0x2
	.byte	0x76
	.byte	0x50
	.byte	0x4
	.set L$set$27,LCFI14-LCFI13
	.long L$set$27
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$28,LCFI15-LCFI14
	.long L$set$28
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$29,LCFI16-LCFI15
	.long L$set$29
	.byte	0xb
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$30,LEFDE13-LASFDE13
	.long L$set$30
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB6-.
	.set L$set$31,LFE6-LFB6
	.quad L$set$31
	.byte	0
	.align 3
LEFDE13:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

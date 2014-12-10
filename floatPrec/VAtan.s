	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB8:
	.text
LHOTB8:
	.align 4,0x90
	.globl __Z6doAtanDv4_f
__Z6doAtanDv4_f:
LFB4098:
	movaps	%xmm0, %xmm2
	movaps	LC0(%rip), %xmm0
	pxor	%xmm3, %xmm3
	cmpltps	%xmm2, %xmm0
	movmskps	%xmm0, %eax
	testl	%eax, %eax
	je	L2
	movaps	LC1(%rip), %xmm1
	movaps	LC2(%rip), %xmm3
	addps	%xmm2, %xmm1
	addps	%xmm2, %xmm3
	divps	%xmm3, %xmm1
	movaps	LC3(%rip), %xmm3
	andps	%xmm0, %xmm3
	blendvps	%xmm0, %xmm1, %xmm2
L2:
	movaps	%xmm2, %xmm0
	movaps	LC4(%rip), %xmm1
	mulps	%xmm2, %xmm0
	mulps	%xmm0, %xmm1
	addps	LC5(%rip), %xmm1
	mulps	%xmm0, %xmm1
	addps	LC6(%rip), %xmm1
	mulps	%xmm0, %xmm1
	addps	LC7(%rip), %xmm1
	mulps	%xmm1, %xmm0
	mulps	%xmm2, %xmm0
	addps	%xmm2, %xmm0
	addps	%xmm3, %xmm0
	ret
LFE4098:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE8:
	.text
LHOTE8:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB9:
	.text
LHOTB9:
	.align 4,0x90
	.globl __Z10computeOnev
__Z10computeOnev:
LFB4100:
	pushq	%r12
LCFI0:
	pushq	%rbp
LCFI1:
	pushq	%rbx
LCFI2:
	xorl	%ebx, %ebx
	subq	$32, %rsp
LCFI3:
	movaps	_va(%rip), %xmm0
	leaq	16(%rsp), %r12
	movq	%rsp, %rbp
	movaps	%xmm0, (%rsp)
L8:
	sqrtss	0(%rbp,%rbx), %xmm0
	ucomiss	%xmm0, %xmm0
	jp	L11
L6:
	movss	%xmm0, (%r12,%rbx)
	addq	$4, %rbx
	cmpq	$16, %rbx
	jne	L8
	movaps	16(%rsp), %xmm0
	movaps	%xmm0, _vb(%rip)
	addq	$32, %rsp
LCFI4:
	popq	%rbx
LCFI5:
	popq	%rbp
LCFI6:
	popq	%r12
LCFI7:
	ret
L11:
LCFI8:
	movss	0(%rbp,%rbx), %xmm0
	call	_sqrtf
	jmp	L6
LFE4100:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE9:
	.text
LHOTE9:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB10:
	.text
LHOTB10:
	.align 4,0x90
	.globl __Z8computeSv
__Z8computeSv:
LFB4101:
	pushq	%r15
LCFI9:
	leaq	_va(%rip), %r15
	pxor	%xmm1, %xmm1
	pushq	%r14
LCFI10:
	leaq	_vb(%rip), %r14
	pushq	%r13
LCFI11:
	pushq	%r12
LCFI12:
	pushq	%rbp
LCFI13:
	xorl	%ebp, %ebp
	pushq	%rbx
LCFI14:
	subq	$56, %rsp
LCFI15:
	leaq	16(%rsp), %r12
	leaq	32(%rsp), %r13
	.align 4,0x90
L16:
	movaps	(%r15,%rbp), %xmm0
	xorl	%ebx, %ebx
	movaps	%xmm0, 16(%rsp)
L15:
	sqrtss	(%r12,%rbx), %xmm0
	ucomiss	%xmm0, %xmm0
	jp	L20
L13:
	movss	%xmm0, 0(%r13,%rbx)
	addq	$4, %rbx
	cmpq	$16, %rbx
	jne	L15
	movaps	32(%rsp), %xmm0
	movaps	%xmm1, 32(%rsp)
	movaps	%xmm0, (%r14,%rbp)
	addq	$16, %rbp
	cmpq	$16384, %rbp
	jne	L16
	addq	$56, %rsp
LCFI16:
	popq	%rbx
LCFI17:
	popq	%rbp
LCFI18:
	popq	%r12
LCFI19:
	popq	%r13
LCFI20:
	popq	%r14
LCFI21:
	popq	%r15
LCFI22:
	ret
L20:
LCFI23:
	movss	(%r12,%rbx), %xmm0
	movaps	%xmm1, (%rsp)
	call	_sqrtf
	movaps	(%rsp), %xmm1
	jmp	L13
LFE4101:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE10:
	.text
LHOTE10:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB11:
	.text
LHOTB11:
	.align 4,0x90
	.globl __Z8computeVv
__Z8computeVv:
LFB4102:
	movaps	LC0(%rip), %xmm7
	xorl	%eax, %eax
	movaps	LC4(%rip), %xmm6
	leaq	_va(%rip), %rsi
	movaps	LC5(%rip), %xmm5
	leaq	_vb(%rip), %rcx
	movaps	LC6(%rip), %xmm4
	movaps	LC7(%rip), %xmm3
	movaps	LC1(%rip), %xmm10
	movaps	LC2(%rip), %xmm9
	movaps	LC3(%rip), %xmm8
	.align 4,0x90
L23:
	movaps	(%rsi,%rax), %xmm11
	movaps	%xmm7, %xmm0
	pxor	%xmm2, %xmm2
	cmpltps	%xmm11, %xmm0
	movmskps	%xmm0, %edx
	testl	%edx, %edx
	je	L22
	movaps	%xmm11, %xmm1
	movaps	%xmm11, %xmm2
	addps	%xmm9, %xmm2
	addps	%xmm10, %xmm1
	divps	%xmm2, %xmm1
	movaps	%xmm0, %xmm2
	andps	%xmm8, %xmm2
	blendvps	%xmm0, %xmm1, %xmm11
L22:
	movaps	%xmm11, %xmm0
	mulps	%xmm11, %xmm0
	movaps	%xmm0, %xmm1
	mulps	%xmm6, %xmm1
	addps	%xmm5, %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm4, %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm3, %xmm1
	mulps	%xmm1, %xmm0
	mulps	%xmm11, %xmm0
	addps	%xmm11, %xmm0
	addps	%xmm2, %xmm0
	movaps	%xmm0, (%rcx,%rax)
	addq	$16, %rax
	cmpq	$16384, %rax
	jne	L23
	ret
LFE4102:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE11:
	.text
LHOTE11:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB20:
	.text
LHOTB20:
	.align 4,0x90
	.globl __Z8computeLv
__Z8computeLv:
LFB4103:
	leaq	_a(%rip), %rcx
	xorl	%eax, %eax
	pxor	%xmm8, %xmm8
	movss	LC16(%rip), %xmm7
	leaq	_b(%rip), %rdx
	movss	LC17(%rip), %xmm6
	movss	LC18(%rip), %xmm5
	movss	LC19(%rip), %xmm4
	movss	LC15(%rip), %xmm3
	movss	LC12(%rip), %xmm10
	jmp	L31
	.align 4,0x90
L37:
	movaps	%xmm0, %xmm1
	addss	%xmm3, %xmm0
	movaps	%xmm10, %xmm2
	subss	%xmm3, %xmm1
	divss	%xmm0, %xmm1
	movaps	%xmm1, %xmm0
L30:
	movaps	%xmm0, %xmm9
	mulss	%xmm0, %xmm9
	movaps	%xmm9, %xmm1
	mulss	%xmm7, %xmm1
	subss	%xmm6, %xmm1
	mulss	%xmm9, %xmm1
	addss	%xmm5, %xmm1
	mulss	%xmm9, %xmm1
	subss	%xmm4, %xmm1
	mulss	%xmm9, %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm1, %xmm0
	addss	%xmm2, %xmm0
	movss	%xmm0, (%rdx,%rax)
	addq	$4, %rax
	cmpq	$16384, %rax
	je	L36
L31:
	movss	(%rcx,%rax), %xmm0
	ucomiss	LC14(%rip), %xmm0
	ja	L37
	ja	L32
	movaps	%xmm8, %xmm2
	jmp	L30
	.align 4,0x90
L36:
	ret
L32:
	movss	LC12(%rip), %xmm2
	jmp	L30
LFE4103:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE20:
	.text
LHOTE20:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB21:
	.text
LHOTB21:
	.align 4,0x90
	.globl __Z8computeAv
__Z8computeAv:
LFB4104:
	movss	LC16(%rip), %xmm7
	xorl	%edx, %edx
	pxor	%xmm8, %xmm8
	pxor	%xmm9, %xmm9
	movss	LC17(%rip), %xmm6
	leaq	-40(%rsp), %rsi
	movss	LC18(%rip), %xmm5
	leaq	-24(%rsp), %rcx
	movss	LC19(%rip), %xmm4
	leaq	_va(%rip), %r8
	movss	LC15(%rip), %xmm3
	leaq	_vb(%rip), %rdi
	movss	LC12(%rip), %xmm11
	.align 4,0x90
L43:
	movaps	(%r8,%rdx), %xmm0
	xorl	%eax, %eax
	movaps	%xmm0, -40(%rsp)
L42:
	movss	(%rsi,%rax), %xmm0
	ucomiss	LC14(%rip), %xmm0
	jbe	L48
	movaps	%xmm0, %xmm1
	addss	%xmm3, %xmm0
	movaps	%xmm11, %xmm2
	subss	%xmm3, %xmm1
	divss	%xmm0, %xmm1
	movaps	%xmm1, %xmm0
L41:
	movaps	%xmm0, %xmm10
	mulss	%xmm0, %xmm10
	movaps	%xmm10, %xmm1
	mulss	%xmm7, %xmm1
	subss	%xmm6, %xmm1
	mulss	%xmm10, %xmm1
	addss	%xmm5, %xmm1
	mulss	%xmm10, %xmm1
	subss	%xmm4, %xmm1
	mulss	%xmm10, %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm1, %xmm0
	addss	%xmm2, %xmm0
	movss	%xmm0, (%rcx,%rax)
	addq	$4, %rax
	cmpq	$16, %rax
	jne	L42
	movaps	-24(%rsp), %xmm0
	movaps	%xmm9, -24(%rsp)
	movaps	%xmm0, (%rdi,%rdx)
	addq	$16, %rdx
	cmpq	$16384, %rdx
	jne	L43
	ret
	.align 4,0x90
L48:
	ja	L44
	movaps	%xmm8, %xmm2
	jmp	L41
L44:
	movss	LC12(%rip), %xmm2
	jmp	L41
LFE4104:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE21:
	.text
LHOTE21:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB22:
	.text
LHOTB22:
	.align 4,0x90
	.globl __Z8computeBv
__Z8computeBv:
LFB4105:
	movss	LC16(%rip), %xmm7
	pxor	%xmm8, %xmm8
	leaq	_vb(%rip), %rdx
	movss	LC17(%rip), %xmm6
	leaq	_va(%rip), %rcx
	movss	LC18(%rip), %xmm5
	leaq	16384+_vb(%rip), %rsi
	movss	LC19(%rip), %xmm4
	movss	LC15(%rip), %xmm3
	movss	LC12(%rip), %xmm10
	.align 4,0x90
L50:
	xorl	%eax, %eax
L55:
	movss	(%rcx,%rax), %xmm0
	ucomiss	LC14(%rip), %xmm0
	jbe	L61
	movaps	%xmm0, %xmm1
	addss	%xmm3, %xmm0
	movaps	%xmm10, %xmm2
	subss	%xmm3, %xmm1
	divss	%xmm0, %xmm1
	movaps	%xmm1, %xmm0
L53:
	movaps	%xmm0, %xmm9
	mulss	%xmm0, %xmm9
	movaps	%xmm9, %xmm1
	mulss	%xmm7, %xmm1
	subss	%xmm6, %xmm1
	mulss	%xmm9, %xmm1
	addss	%xmm5, %xmm1
	mulss	%xmm9, %xmm1
	subss	%xmm4, %xmm1
	mulss	%xmm9, %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm1, %xmm0
	addss	%xmm2, %xmm0
	movss	%xmm0, (%rdx,%rax)
	addq	$4, %rax
	cmpq	$16, %rax
	jne	L55
	addq	$16, %rdx
	addq	$16, %rcx
	cmpq	%rsi, %rdx
	jne	L50
	ret
	.align 4,0x90
L61:
	ja	L63
	movaps	%xmm8, %xmm2
	jmp	L53
L63:
	movss	LC12(%rip), %xmm2
	jmp	L53
LFE4105:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE22:
	.text
LHOTE22:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB24:
	.text
LHOTB24:
	.align 4,0x90
	.globl __Z5fillRv
__Z5fillRv:
LFB5253:
	movss	_rgen(%rip), %xmm2
	pushq	%rbp
LCFI24:
	leaq	_va(%rip), %r9
	movl	$-1727483681, %edx
	movss	4+_rgen(%rip), %xmm1
	pushq	%rbx
LCFI25:
	leaq	16384+_va(%rip), %r11
	pxor	%xmm4, %xmm4
	movq	2496+_eng(%rip), %rbp
	leaq	-24(%rsp), %r8
	pxor	%xmm5, %xmm5
	subss	%xmm2, %xmm1
	leaq	_eng(%rip), %r10
	movss	LC23(%rip), %xmm3
	leaq	908+_eng(%rip), %rsi
	leaq	2492+_eng(%rip), %rcx
L75:
	xorl	%edi, %edi
L74:
	cmpq	$623, %rbp
	ja	L65
	movl	(%r10,%rbp,4), %eax
	addq	$1, %rbp
L66:
	movl	%eax, %ebx
	pxor	%xmm0, %xmm0
	shrl	$11, %ebx
	xorl	%ebx, %eax
	movl	%eax, %ebx
	sall	$7, %ebx
	andl	$-1658038656, %ebx
	xorl	%eax, %ebx
	movl	%ebx, %eax
	sall	$15, %eax
	andl	$-272236544, %eax
	xorl	%ebx, %eax
	movl	%eax, %ebx
	shrl	$18, %ebx
	xorl	%ebx, %eax
	cvtsi2ssq	%rax, %xmm0
	addss	%xmm4, %xmm0
	mulss	%xmm3, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm2, %xmm0
	movss	%xmm0, (%r8,%rdi)
	addq	$4, %rdi
	cmpq	$16, %rdi
	jne	L74
	movaps	-24(%rsp), %xmm0
	addq	$16, %r9
	movaps	%xmm5, -24(%rsp)
	movaps	%xmm0, -16(%r9)
	cmpq	%r11, %r9
	jne	L75
	movq	%rbp, 2496+_eng(%rip)
	popq	%rbx
LCFI26:
	popq	%rbp
LCFI27:
	ret
	.align 4,0x90
L65:
LCFI28:
	leaq	_eng(%rip), %rbx
	.align 4,0x90
L68:
	movl	(%rbx), %eax
	movl	4(%rbx), %ebp
	andl	$-2147483648, %eax
	andl	$2147483647, %ebp
	orl	%ebp, %eax
	movl	%eax, %ebp
	shrl	%ebp
	xorl	1588(%rbx), %ebp
	andl	$1, %eax
	cmovne	%edx, %eax
	addq	$4, %rbx
	xorl	%ebp, %eax
	movl	%eax, -4(%rbx)
	cmpq	%rsi, %rbx
	jne	L68
	leaq	908+_eng(%rip), %rbx
	.align 4,0x90
L70:
	movl	(%rbx), %eax
	movl	4(%rbx), %ebp
	andl	$-2147483648, %eax
	andl	$2147483647, %ebp
	orl	%ebp, %eax
	movl	%eax, %ebp
	shrl	%ebp
	xorl	-908(%rbx), %ebp
	andl	$1, %eax
	cmovne	%edx, %eax
	addq	$4, %rbx
	xorl	%ebp, %eax
	movl	%eax, -4(%rbx)
	cmpq	%rcx, %rbx
	jne	L70
	movl	_eng(%rip), %eax
	movl	2492+_eng(%rip), %ebx
	movl	%eax, %ebp
	andl	$2147483647, %ebp
	andl	$-2147483648, %ebx
	orl	%ebp, %ebx
	movl	%ebx, %ebp
	shrl	%ebp
	xorl	1584+_eng(%rip), %ebp
	andl	$1, %ebx
	cmovne	%edx, %ebx
	xorl	%ebp, %ebx
	movl	$1, %ebp
	movl	%ebx, 2492+_eng(%rip)
	jmp	L66
LFE5253:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE24:
	.text
LHOTE24:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB26:
	.text
LHOTB26:
	.align 4,0x90
	.globl __Z5fillOv
__Z5fillOv:
LFB5257:
	movsd	LC25(%rip), %xmm1
	leaq	_va(%rip), %rax
	pxor	%xmm0, %xmm0
	leaq	16384+_va(%rip), %rcx
	.align 4,0x90
L88:
	leaq	16(%rax), %rdx
L89:
	cvtss2sd	%xmm0, %xmm0
	addsd	%xmm1, %xmm0
	addq	$4, %rax
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -4(%rax)
	cmpq	%rdx, %rax
	jne	L89
	cmpq	%rcx, %rax
	jne	L88
	ret
LFE5257:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE26:
	.text
LHOTE26:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB28:
	.text
LHOTB28:
	.align 4,0x90
	.globl __Z5fillWv
__Z5fillWv:
LFB5258:
	movss	LC27(%rip), %xmm0
	leaq	_va(%rip), %rax
	pxor	%xmm1, %xmm1
	movsd	LC25(%rip), %xmm2
	leaq	16384+_va(%rip), %rdx
	.align 4,0x90
L93:
	cvtss2sd	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm0
	addsd	%xmm2, %xmm1
	addq	$16, %rax
	subsd	%xmm2, %xmm0
	cvtsd2ss	%xmm1, %xmm1
	movss	%xmm1, -16(%rax)
	cvtss2sd	%xmm1, %xmm1
	addsd	%xmm2, %xmm1
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -12(%rax)
	cvtss2sd	%xmm0, %xmm0
	subsd	%xmm2, %xmm0
	cvtsd2ss	%xmm1, %xmm1
	movss	%xmm1, -8(%rax)
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -4(%rax)
	cmpq	%rdx, %rax
	jne	L93
	ret
LFE5258:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE28:
	.text
LHOTE28:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB29:
	.text
LHOTB29:
	.align 4,0x90
	.globl __Z3sumv
__Z3sumv:
LFB5259:
	leaq	_vb(%rip), %rax
	pxor	%xmm0, %xmm0
	leaq	16384+_vb(%rip), %rdx
	.align 4,0x90
L96:
	addss	(%rax), %xmm0
	addq	$16, %rax
	subss	-12(%rax), %xmm0
	addss	-8(%rax), %xmm0
	subss	-4(%rax), %xmm0
	cmpq	%rdx, %rax
	jne	L96
	ret
LFE5259:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE29:
	.text
LHOTE29:
	.cstring
LC30:
	.ascii " \0"
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB32:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTB32:
	.align 4
	.globl _main
_main:
LFB5261:
	pushq	%r15
LCFI29:
	pushq	%r14
LCFI30:
	pushq	%r13
LCFI31:
	pushq	%r12
LCFI32:
	xorl	%r12d, %r12d
	pushq	%rbp
LCFI33:
	xorl	%ebp, %ebp
	pushq	%rbx
LCFI34:
	xorl	%ebx, %ebx
	subq	$40, %rsp
LCFI35:
	call	__Z5fillRv
	call	__Z8computeVv
	movl	$10000, %r10d
	movsd	LC25(%rip), %xmm2
	movl	$0x00000000, 4(%rsp)
	movl	$0x00000000, (%rsp)
	leaq	16384+_va(%rip), %r9
	pxor	%xmm12, %xmm12
L99:
	leaq	_va(%rip), %r14
	pxor	%xmm1, %xmm1
	leaq	16384+_va(%rip), %r13
	movq	%r14, %rax
	.align 4
L103:
	leaq	16(%rax), %rdx
L100:
	cvtss2sd	%xmm1, %xmm1
	addsd	%xmm2, %xmm1
	addq	$4, %rax
	cvtsd2ss	%xmm1, %xmm1
	movss	%xmm1, -4(%rax)
	cmpq	%rdx, %rax
	jne	L100
	cmpq	%r9, %rax
	jne	L103
	movsd	%xmm2, 8(%rsp)
	rdtscp
	movq	%rax, %rdi
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rdi
	call	__Z8computeVv
	rdtscp
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rax
	subq	%rdi, %rax
	addq	%rax, %r12
	call	__Z3sumv
	addss	%xmm0, %xmm12
	rdtscp
	movq	%rax, %r11
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %r11
	call	__Z8computeAv
	rdtscp
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rax
	subq	%r11, %rax
	addq	%rax, %rbp
	call	__Z3sumv
	addss	(%rsp), %xmm0
	movss	%xmm0, (%rsp)
	rdtscp
	movq	%rax, %rdi
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rdi
	call	__Z8computeBv
	rdtscp
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rax
	subq	%rdi, %rax
	addq	%rax, %rbx
	call	__Z3sumv
	addss	4(%rsp), %xmm0
	subl	$1, %r10d
	movsd	8(%rsp), %xmm2
	movss	%xmm0, 4(%rsp)
	jne	L99
	movq	__ZSt4cout@GOTPCREL(%rip), %r15
	movsd	%xmm2, 8(%rsp)
	pxor	%xmm0, %xmm0
	cvtss2sd	%xmm12, %xmm0
	movq	%r15, %rdi
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC30(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%r12, %xmm0
	movq	%rax, %rdi
	divsd	LC31(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%r15, %rdi
	pxor	%xmm0, %xmm0
	cvtss2sd	(%rsp), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC30(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rbp, %xmm0
	movq	%rax, %rdi
	divsd	LC31(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%r15, %rdi
	pxor	%xmm0, %xmm0
	cvtss2sd	4(%rsp), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC30(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rbx, %xmm0
	movq	%rax, %rdi
	divsd	LC31(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movsd	8(%rsp), %xmm2
	pxor	%xmm0, %xmm0
	leaq	_va(%rip), %rax
L104:
	leaq	16(%rax), %rdx
L105:
	cvtss2sd	%xmm0, %xmm0
	addsd	%xmm2, %xmm0
	addq	$4, %rax
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -4(%rax)
	cmpq	%rdx, %rax
	jne	L105
	cmpq	%rax, %r13
	jne	L104
	movsd	%xmm2, 16(%rsp)
	xorl	%ebx, %ebx
	xorl	%ebp, %ebp
	xorl	%r12d, %r12d
	call	__Z8computeVv
	movsd	16(%rsp), %xmm2
	movl	$10000, %r8d
	movl	$0x00000000, 8(%rsp)
	movl	$0x00000000, 4(%rsp)
	movl	$0x00000000, (%rsp)
L107:
	movsd	%xmm2, 24(%rsp)
	movl	%r8d, 16(%rsp)
	call	__Z5fillRv
	rdtscp
	movq	%rax, %rdi
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rdi
	call	__Z8computeVv
	rdtscp
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rax
	addq	%rax, %r12
	subq	%rdi, %r12
	call	__Z3sumv
	addss	(%rsp), %xmm0
	movss	%xmm0, (%rsp)
	rdtscp
	movq	%rax, %r9
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %r9
	call	__Z8computeAv
	rdtscp
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rax
	addq	%rax, %rbp
	subq	%r9, %rbp
	call	__Z3sumv
	addss	4(%rsp), %xmm0
	movl	$4096, %edx
	leaq	_va(%rip), %rsi
	leaq	_a(%rip), %rdi
	movss	%xmm0, 4(%rsp)
	call	_memcpy
	rdtscp
	movq	%rax, %rsi
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rsi
	call	__Z8computeLv
	rdtscp
	leaq	_vb(%rip), %rdi
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rax
	movl	$4096, %edx
	addq	%rax, %rbx
	subq	%rsi, %rbx
	leaq	_b(%rip), %rsi
	call	_memcpy
	call	__Z3sumv
	movl	16(%rsp), %r8d
	addss	8(%rsp), %xmm0
	movsd	24(%rsp), %xmm2
	subl	$1, %r8d
	movss	%xmm0, 8(%rsp)
	jne	L107
	movq	%r15, %rdi
	movsd	%xmm2, 16(%rsp)
	pxor	%xmm0, %xmm0
	cvtss2sd	(%rsp), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC30(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%r12, %xmm0
	movq	%rax, %rdi
	divsd	LC31(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%r15, %rdi
	pxor	%xmm0, %xmm0
	cvtss2sd	4(%rsp), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC30(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rbp, %xmm0
	movq	%rax, %rdi
	divsd	LC31(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%r15, %rdi
	pxor	%xmm0, %xmm0
	cvtss2sd	8(%rsp), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC30(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rbx, %xmm0
	movq	%rax, %rdi
	divsd	LC31(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movsd	16(%rsp), %xmm2
	pxor	%xmm0, %xmm0
L108:
	leaq	16(%r14), %rax
L109:
	cvtss2sd	%xmm0, %xmm0
	addsd	%xmm2, %xmm0
	addq	$4, %r14
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -4(%r14)
	cmpq	%rax, %r14
	jne	L109
	cmpq	%r14, %r13
	jne	L108
	call	__Z8computeVv
	xorl	%ebx, %ebx
	pxor	%xmm1, %xmm1
	xorl	%ebp, %ebp
	movl	$10000, %r9d
	movss	%xmm1, (%rsp)
	movaps	%xmm1, %xmm12
	xorl	%r12d, %r12d
L111:
	movss	%xmm1, 4(%rsp)
	call	__Z5fillWv
	rdtscp
	movq	%rax, %rdi
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rdi
	call	__Z8computeVv
	rdtscp
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rax
	addq	%rax, %r12
	subq	%rdi, %r12
	call	__Z3sumv
	addss	%xmm0, %xmm12
	rdtscp
	movq	%rax, %r10
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %r10
	call	__Z8computeAv
	rdtscp
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rax
	addq	%rax, %rbp
	subq	%r10, %rbp
	call	__Z3sumv
	addss	(%rsp), %xmm0
	movss	%xmm0, (%rsp)
	rdtscp
	movq	%rax, %rdi
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rdi
	call	__Z8computeBv
	rdtscp
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rax
	addq	%rbx, %rax
	subq	%rdi, %rax
	movq	%rax, %rbx
	call	__Z3sumv
	movss	4(%rsp), %xmm1
	subl	$1, %r9d
	addss	%xmm0, %xmm1
	jne	L111
	movq	%r15, %rdi
	movss	%xmm1, 4(%rsp)
	pxor	%xmm0, %xmm0
	cvtss2sd	%xmm12, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC30(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%r12, %xmm0
	movq	%rax, %rdi
	divsd	LC31(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%r15, %rdi
	pxor	%xmm0, %xmm0
	cvtss2sd	(%rsp), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC30(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rbp, %xmm0
	movq	%rax, %rdi
	divsd	LC31(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movss	4(%rsp), %xmm1
	movq	%r15, %rdi
	pxor	%xmm0, %xmm0
	cvtss2sd	%xmm1, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC30(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rbx, %xmm0
	movq	%rax, %rdi
	divsd	LC31(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	addq	$40, %rsp
LCFI36:
	xorl	%eax, %eax
	popq	%rbx
LCFI37:
	popq	%rbp
LCFI38:
	popq	%r12
LCFI39:
	popq	%r13
LCFI40:
	popq	%r14
LCFI41:
	popq	%r15
LCFI42:
	ret
LFE5261:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE32:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTE32:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB33:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTB33:
	.align 4
__GLOBAL__sub_I_VAtan.cpp:
LFB5674:
	leaq	__ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
LCFI43:
	call	__ZNSt8ios_base4InitC1Ev
	movq	__ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	leaq	___dso_handle(%rip), %rdx
	leaq	__ZStL8__ioinit(%rip), %rsi
	call	___cxa_atexit
	movl	$5489, %eax
	movl	$1, %ecx
	movl	$5489, _eng(%rip)
	leaq	_eng(%rip), %r8
	movl	$440509467, %edi
	jmp	L120
	.align 4
L124:
	leaq	-4+_eng(%rip), %rax
	movl	4(%rax,%r9), %eax
L120:
	leaq	0(,%rcx,4), %r9
	movl	%eax, %esi
	movl	%ecx, %edx
	shrl	$30, %esi
	shrl	$4, %edx
	xorl	%esi, %eax
	imull	$1812433253, %eax, %esi
	movl	%edx, %eax
	mull	%edi
	movl	%ecx, %eax
	shrl	$2, %edx
	imull	$624, %edx, %edx
	subl	%edx, %eax
	addl	%eax, %esi
	movl	%esi, (%r8,%rcx,4)
	addq	$1, %rcx
	cmpq	$624, %rcx
	jne	L124
	movl	$5489, %eax
	movl	$1, %ecx
	movl	$440509467, %edi
	movq	$624, 2496+_eng(%rip)
	movl	$5489, _eng2(%rip)
	leaq	_eng2(%rip), %r8
	jmp	L122
	.align 4
L125:
	leaq	-4+_eng2(%rip), %rax
	movl	4(%rax,%r9), %eax
L122:
	leaq	0(,%rcx,4), %r9
	movl	%eax, %esi
	movl	%ecx, %edx
	shrl	$30, %esi
	shrl	$4, %edx
	xorl	%esi, %eax
	imull	$1812433253, %eax, %esi
	movl	%edx, %eax
	mull	%edi
	movl	%ecx, %eax
	shrl	$2, %edx
	imull	$624, %edx, %edx
	subl	%edx, %eax
	addl	%eax, %esi
	movl	%esi, (%r8,%rcx,4)
	addq	$1, %rcx
	cmpq	$624, %rcx
	jne	L125
	movq	$624, 2496+_eng2(%rip)
	movl	$0x00000000, _rgen(%rip)
	movl	$0x3f800000, 4+_rgen(%rip)
	addq	$8, %rsp
LCFI44:
	ret
LFE5674:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE33:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTE33:
	.globl _taux
	.zerofill __DATA,__pu_bss2,_taux,4,2
	.globl _rgen
	.zerofill __DATA,__pu_bss2,_rgen,8,2
	.globl _eng2
	.zerofill __DATA,__pu_bss6,_eng2,2504,6
	.globl _eng
	.zerofill __DATA,__pu_bss6,_eng,2504,6
	.globl _b
	.zerofill __DATA,__pu_bss6,_b,16384,6
	.globl _a
	.zerofill __DATA,__pu_bss6,_a,16384,6
	.globl _vc
	.zerofill __DATA,__pu_bss6,_vc,16384,6
	.globl _vb
	.zerofill __DATA,__pu_bss6,_vb,16384,6
	.globl _va
	.zerofill __DATA,__pu_bss6,_va,16384,6
	.static_data
__ZStL8__ioinit:
	.space	1
	.literal16
	.align 4
LC0:
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.align 4
LC1:
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.align 4
LC2:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 4
LC3:
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.align 4
LC4:
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.align 4
LC5:
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.align 4
LC6:
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.align 4
LC7:
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.literal4
	.align 2
LC12:
	.long	1061752795
	.align 2
LC14:
	.long	1054086093
	.align 2
LC15:
	.long	1065353216
	.align 2
LC16:
	.long	1034219729
	.align 2
LC17:
	.long	1041111941
	.align 2
LC18:
	.long	1045205599
	.align 2
LC19:
	.long	1051372074
	.align 2
LC23:
	.long	796917760
	.literal8
	.align 3
LC25:
	.long	2576980378
	.long	1059690905
	.literal4
	.align 2
LC27:
	.long	1061997773
	.literal8
	.align 3
LC31:
	.long	0
	.long	1086556160
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
	.quad	LFB4098-.
	.set L$set$2,LFE4098-LFB4098
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB4100-.
	.set L$set$4,LFE4100-LFB4100
	.quad L$set$4
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI0-LFB4100
	.long L$set$5
	.byte	0xe
	.byte	0x10
	.byte	0x8c
	.byte	0x2
	.byte	0x4
	.set L$set$6,LCFI1-LCFI0
	.long L$set$6
	.byte	0xe
	.byte	0x18
	.byte	0x86
	.byte	0x3
	.byte	0x4
	.set L$set$7,LCFI2-LCFI1
	.long L$set$7
	.byte	0xe
	.byte	0x20
	.byte	0x83
	.byte	0x4
	.byte	0x4
	.set L$set$8,LCFI3-LCFI2
	.long L$set$8
	.byte	0xe
	.byte	0x40
	.byte	0x4
	.set L$set$9,LCFI4-LCFI3
	.long L$set$9
	.byte	0xa
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$10,LCFI5-LCFI4
	.long L$set$10
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$11,LCFI6-LCFI5
	.long L$set$11
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$12,LCFI7-LCFI6
	.long L$set$12
	.byte	0xe
	.byte	0x8
	.byte	0x4
	.set L$set$13,LCFI8-LCFI7
	.long L$set$13
	.byte	0xb
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$14,LEFDE5-LASFDE5
	.long L$set$14
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB4101-.
	.set L$set$15,LFE4101-LFB4101
	.quad L$set$15
	.byte	0
	.byte	0x4
	.set L$set$16,LCFI9-LFB4101
	.long L$set$16
	.byte	0xe
	.byte	0x10
	.byte	0x8f
	.byte	0x2
	.byte	0x4
	.set L$set$17,LCFI10-LCFI9
	.long L$set$17
	.byte	0xe
	.byte	0x18
	.byte	0x8e
	.byte	0x3
	.byte	0x4
	.set L$set$18,LCFI11-LCFI10
	.long L$set$18
	.byte	0xe
	.byte	0x20
	.byte	0x8d
	.byte	0x4
	.byte	0x4
	.set L$set$19,LCFI12-LCFI11
	.long L$set$19
	.byte	0xe
	.byte	0x28
	.byte	0x8c
	.byte	0x5
	.byte	0x4
	.set L$set$20,LCFI13-LCFI12
	.long L$set$20
	.byte	0xe
	.byte	0x30
	.byte	0x86
	.byte	0x6
	.byte	0x4
	.set L$set$21,LCFI14-LCFI13
	.long L$set$21
	.byte	0xe
	.byte	0x38
	.byte	0x83
	.byte	0x7
	.byte	0x4
	.set L$set$22,LCFI15-LCFI14
	.long L$set$22
	.byte	0xe
	.byte	0x70
	.byte	0x4
	.set L$set$23,LCFI16-LCFI15
	.long L$set$23
	.byte	0xa
	.byte	0xe
	.byte	0x38
	.byte	0x4
	.set L$set$24,LCFI17-LCFI16
	.long L$set$24
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$25,LCFI18-LCFI17
	.long L$set$25
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$26,LCFI19-LCFI18
	.long L$set$26
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$27,LCFI20-LCFI19
	.long L$set$27
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$28,LCFI21-LCFI20
	.long L$set$28
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$29,LCFI22-LCFI21
	.long L$set$29
	.byte	0xe
	.byte	0x8
	.byte	0x4
	.set L$set$30,LCFI23-LCFI22
	.long L$set$30
	.byte	0xb
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$31,LEFDE7-LASFDE7
	.long L$set$31
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB4102-.
	.set L$set$32,LFE4102-LFB4102
	.quad L$set$32
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$33,LEFDE9-LASFDE9
	.long L$set$33
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB4103-.
	.set L$set$34,LFE4103-LFB4103
	.quad L$set$34
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$35,LEFDE11-LASFDE11
	.long L$set$35
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB4104-.
	.set L$set$36,LFE4104-LFB4104
	.quad L$set$36
	.byte	0
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$37,LEFDE13-LASFDE13
	.long L$set$37
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB4105-.
	.set L$set$38,LFE4105-LFB4105
	.quad L$set$38
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$39,LEFDE15-LASFDE15
	.long L$set$39
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB5253-.
	.set L$set$40,LFE5253-LFB5253
	.quad L$set$40
	.byte	0
	.byte	0x4
	.set L$set$41,LCFI24-LFB5253
	.long L$set$41
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$42,LCFI25-LCFI24
	.long L$set$42
	.byte	0xe
	.byte	0x18
	.byte	0x83
	.byte	0x3
	.byte	0x4
	.set L$set$43,LCFI26-LCFI25
	.long L$set$43
	.byte	0xa
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$44,LCFI27-LCFI26
	.long L$set$44
	.byte	0xe
	.byte	0x8
	.byte	0x4
	.set L$set$45,LCFI28-LCFI27
	.long L$set$45
	.byte	0xb
	.align 3
LEFDE15:
LSFDE17:
	.set L$set$46,LEFDE17-LASFDE17
	.long L$set$46
LASFDE17:
	.long	LASFDE17-EH_frame1
	.quad	LFB5257-.
	.set L$set$47,LFE5257-LFB5257
	.quad L$set$47
	.byte	0
	.align 3
LEFDE17:
LSFDE19:
	.set L$set$48,LEFDE19-LASFDE19
	.long L$set$48
LASFDE19:
	.long	LASFDE19-EH_frame1
	.quad	LFB5258-.
	.set L$set$49,LFE5258-LFB5258
	.quad L$set$49
	.byte	0
	.align 3
LEFDE19:
LSFDE21:
	.set L$set$50,LEFDE21-LASFDE21
	.long L$set$50
LASFDE21:
	.long	LASFDE21-EH_frame1
	.quad	LFB5259-.
	.set L$set$51,LFE5259-LFB5259
	.quad L$set$51
	.byte	0
	.align 3
LEFDE21:
LSFDE23:
	.set L$set$52,LEFDE23-LASFDE23
	.long L$set$52
LASFDE23:
	.long	LASFDE23-EH_frame1
	.quad	LFB5261-.
	.set L$set$53,LFE5261-LFB5261
	.quad L$set$53
	.byte	0
	.byte	0x4
	.set L$set$54,LCFI29-LFB5261
	.long L$set$54
	.byte	0xe
	.byte	0x10
	.byte	0x8f
	.byte	0x2
	.byte	0x4
	.set L$set$55,LCFI30-LCFI29
	.long L$set$55
	.byte	0xe
	.byte	0x18
	.byte	0x8e
	.byte	0x3
	.byte	0x4
	.set L$set$56,LCFI31-LCFI30
	.long L$set$56
	.byte	0xe
	.byte	0x20
	.byte	0x8d
	.byte	0x4
	.byte	0x4
	.set L$set$57,LCFI32-LCFI31
	.long L$set$57
	.byte	0xe
	.byte	0x28
	.byte	0x8c
	.byte	0x5
	.byte	0x4
	.set L$set$58,LCFI33-LCFI32
	.long L$set$58
	.byte	0xe
	.byte	0x30
	.byte	0x86
	.byte	0x6
	.byte	0x4
	.set L$set$59,LCFI34-LCFI33
	.long L$set$59
	.byte	0xe
	.byte	0x38
	.byte	0x83
	.byte	0x7
	.byte	0x4
	.set L$set$60,LCFI35-LCFI34
	.long L$set$60
	.byte	0xe
	.byte	0x60
	.byte	0x4
	.set L$set$61,LCFI36-LCFI35
	.long L$set$61
	.byte	0xe
	.byte	0x38
	.byte	0x4
	.set L$set$62,LCFI37-LCFI36
	.long L$set$62
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$63,LCFI38-LCFI37
	.long L$set$63
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$64,LCFI39-LCFI38
	.long L$set$64
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$65,LCFI40-LCFI39
	.long L$set$65
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$66,LCFI41-LCFI40
	.long L$set$66
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$67,LCFI42-LCFI41
	.long L$set$67
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE23:
LSFDE25:
	.set L$set$68,LEFDE25-LASFDE25
	.long L$set$68
LASFDE25:
	.long	LASFDE25-EH_frame1
	.quad	LFB5674-.
	.set L$set$69,LFE5674-LFB5674
	.quad L$set$69
	.byte	0
	.byte	0x4
	.set L$set$70,LCFI43-LFB5674
	.long L$set$70
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$71,LCFI44-LCFI43
	.long L$set$71
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE25:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_VAtan.cpp
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

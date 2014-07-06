	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB1:
	.text
LHOTB1:
	.align 4,0x90
	.globl __Z3foodd
__Z3foodd:
LFB0:
	vmovapd	%xmm1, %xmm3
	vmovapd	%xmm0, %xmm2
	xorl	%edx, %edx
	movl	$100, %eax
	.align 4,0x90
L4:
	vmulsd	%xmm2, %xmm2, %xmm2
	vmulsd	%xmm3, %xmm3, %xmm4
	vaddsd	%xmm4, %xmm2, %xmm5
	vucomisd	LC0(%rip), %xmm5
	jbe	L2
	cmpq	%rdx, %rax
	cmovg	%rdx, %rax
L2:
	vaddsd	%xmm2, %xmm0, %xmm2
	addq	$1, %rdx
	vsubsd	%xmm4, %xmm2, %xmm2
	vaddsd	%xmm2, %xmm2, %xmm4
	vfmadd132sd	%xmm4, %xmm1, %xmm3
	cmpq	$100, %rdx
	jne	L4
	ret
LFE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE1:
	.text
LHOTE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB2:
	.text
LHOTB2:
	.align 4,0x90
	.globl __Z3barff
__Z3barff:
LFB1:
	vxorpd	%xmm5, %xmm5, %xmm5
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vcvtss2sd	%xmm0, %xmm5, %xmm5
	vmovapd	%xmm1, %xmm2
	vmovapd	%xmm5, %xmm0
	xorl	%edx, %edx
	movl	$100, %eax
	.align 4,0x90
L14:
	vmulsd	%xmm0, %xmm0, %xmm0
	vmulsd	%xmm2, %xmm2, %xmm3
	vaddsd	%xmm3, %xmm0, %xmm4
	vucomisd	LC0(%rip), %xmm4
	jbe	L12
	cmpl	%edx, %eax
	cmovg	%edx, %eax
L12:
	vaddsd	%xmm0, %xmm5, %xmm0
	addl	$1, %edx
	vsubsd	%xmm3, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm3
	vfmadd132sd	%xmm3, %xmm1, %xmm2
	cmpl	$100, %edx
	jne	L14
	ret
LFE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE2:
	.text
LHOTE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB3:
	.text
LHOTB3:
	.align 4,0x90
	.globl __ZGVbN8vv__Z3barff
__ZGVbN8vv__Z3barff:
LFB2:
	vmovups	%xmm0, -72(%rsp)
	leaq	-72(%rsp), %r8
	xorl	%ecx, %ecx
	vmovups	%xmm1, -56(%rsp)
	leaq	-40(%rsp), %rsi
	vmovups	%xmm2, -40(%rsp)
	vmovups	%xmm3, -24(%rsp)
L24:
	vxorpd	%xmm4, %xmm4, %xmm4
	vxorpd	%xmm5, %xmm5, %xmm5
	vcvtss2sd	(%r8,%rcx), %xmm4, %xmm4
	vcvtss2sd	(%rsi,%rcx), %xmm5, %xmm5
	vmovapd	%xmm4, %xmm0
	vmovapd	%xmm5, %xmm1
	movl	$100, %edx
	xorl	%eax, %eax
	.align 4,0x90
L23:
	vmulsd	%xmm0, %xmm0, %xmm0
	vmulsd	%xmm1, %xmm1, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm3
	vucomisd	LC0(%rip), %xmm3
	jbe	L21
	cmpl	%edx, %eax
	cmovl	%eax, %edx
L21:
	vaddsd	%xmm0, %xmm4, %xmm0
	addl	$1, %eax
	vsubsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm2
	vfmadd132sd	%xmm2, %xmm5, %xmm1
	cmpl	$100, %eax
	jne	L23
	movl	%edx, (%rcx,%rdi)
	addq	$4, %rcx
	cmpq	$32, %rcx
	jne	L24
	ret
LFE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE3:
	.text
LHOTE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB4:
	.text
LHOTB4:
	.align 4,0x90
	.globl __ZGVcN8vv__Z3barff
__ZGVcN8vv__Z3barff:
LFB3:
	leaq	8(%rsp), %r10
LCFI0:
	andq	$-32, %rsp
	xorl	%ecx, %ecx
	pushq	-8(%r10)
	pushq	%rbp
LCFI1:
	movq	%rsp, %rbp
	pushq	%r10
LCFI2:
	leaq	-80(%rbp), %r8
	leaq	-48(%rbp), %rsi
	vmovups	%ymm0, -80(%rbp)
	vmovups	%ymm1, -48(%rbp)
L35:
	vxorpd	%xmm4, %xmm4, %xmm4
	vxorpd	%xmm5, %xmm5, %xmm5
	vcvtss2sd	(%r8,%rcx), %xmm4, %xmm4
	vcvtss2sd	(%rsi,%rcx), %xmm5, %xmm5
	vmovapd	%xmm4, %xmm0
	vmovapd	%xmm5, %xmm1
	movl	$100, %edx
	xorl	%eax, %eax
	.align 4,0x90
L34:
	vmulsd	%xmm0, %xmm0, %xmm0
	vmulsd	%xmm1, %xmm1, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm3
	vucomisd	LC0(%rip), %xmm3
	jbe	L32
	cmpl	%edx, %eax
	cmovl	%eax, %edx
L32:
	vaddsd	%xmm0, %xmm4, %xmm0
	addl	$1, %eax
	vsubsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm2
	vfmadd132sd	%xmm2, %xmm5, %xmm1
	cmpl	$100, %eax
	jne	L34
	movl	%edx, (%rcx,%rdi)
	addq	$4, %rcx
	cmpq	$32, %rcx
	jne	L35
	vzeroupper
	popq	%r10
LCFI3:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI4:
	ret
LFE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE4:
	.text
LHOTE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB5:
	.text
LHOTB5:
	.align 4,0x90
	.globl __ZGVdN8vv__Z3barff
__ZGVdN8vv__Z3barff:
LFB4:
	leaq	8(%rsp), %r10
LCFI5:
	andq	$-32, %rsp
	xorl	%ecx, %ecx
	pushq	-8(%r10)
	pushq	%rbp
LCFI6:
	movq	%rsp, %rbp
	pushq	%r10
LCFI7:
	leaq	-80(%rbp), %r8
	leaq	-48(%rbp), %rdi
	vmovaps	%ymm0, -80(%rbp)
	vmovaps	%ymm1, -48(%rbp)
	leaq	-112(%rbp), %rsi
L47:
	vxorpd	%xmm4, %xmm4, %xmm4
	vxorpd	%xmm5, %xmm5, %xmm5
	vcvtss2sd	(%r8,%rcx), %xmm4, %xmm4
	vcvtss2sd	(%rdi,%rcx), %xmm5, %xmm5
	vmovapd	%xmm4, %xmm0
	vmovapd	%xmm5, %xmm1
	movl	$100, %edx
	xorl	%eax, %eax
	.align 4,0x90
L46:
	vmulsd	%xmm0, %xmm0, %xmm0
	vmulsd	%xmm1, %xmm1, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm3
	vucomisd	LC0(%rip), %xmm3
	jbe	L44
	cmpl	%edx, %eax
	cmovl	%eax, %edx
L44:
	vaddsd	%xmm0, %xmm4, %xmm0
	addl	$1, %eax
	vsubsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm2
	vfmadd132sd	%xmm2, %xmm5, %xmm1
	cmpl	$100, %eax
	jne	L46
	movl	%edx, (%rsi,%rcx)
	addq	$4, %rcx
	cmpq	$32, %rcx
	jne	L47
	vmovdqa	-112(%rbp), %ymm0
	popq	%r10
LCFI8:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI9:
	ret
LFE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE5:
	.text
LHOTE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB6:
	.text
LHOTB6:
	.align 4,0x90
	.globl __ZGVbN4vv__Z3foodd
__ZGVbN4vv__Z3foodd:
LFB5:
	vmovups	%xmm0, -72(%rsp)
	leaq	-40(%rsp), %r8
	xorl	%ecx, %ecx
	vmovups	%xmm1, -56(%rsp)
	leaq	-72(%rsp), %rsi
	vmovups	%xmm2, -40(%rsp)
	vmovups	%xmm3, -24(%rsp)
L59:
	vmovsd	(%r8,%rcx), %xmm5
	movl	$100, %edx
	xorl	%eax, %eax
	vmovsd	(%rsi,%rcx), %xmm4
	vmovapd	%xmm5, %xmm1
	vmovapd	%xmm4, %xmm0
	.align 4,0x90
L58:
	vmulsd	%xmm0, %xmm0, %xmm0
	vmulsd	%xmm1, %xmm1, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm3
	vucomisd	LC0(%rip), %xmm3
	jbe	L56
	cmpq	%rdx, %rax
	cmovl	%rax, %rdx
L56:
	vaddsd	%xmm4, %xmm0, %xmm0
	addq	$1, %rax
	vsubsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm2
	vfmadd132sd	%xmm2, %xmm5, %xmm1
	cmpq	$100, %rax
	jne	L58
	movq	%rdx, (%rcx,%rdi)
	addq	$8, %rcx
	cmpq	$32, %rcx
	jne	L59
	ret
LFE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE6:
	.text
LHOTE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB7:
	.text
LHOTB7:
	.align 4,0x90
	.globl __ZGVcN4vv__Z3foodd
__ZGVcN4vv__Z3foodd:
LFB6:
	leaq	8(%rsp), %r10
LCFI10:
	andq	$-32, %rsp
	xorl	%ecx, %ecx
	pushq	-8(%r10)
	pushq	%rbp
LCFI11:
	movq	%rsp, %rbp
	pushq	%r10
LCFI12:
	leaq	-48(%rbp), %r8
	leaq	-80(%rbp), %rsi
	vmovupd	%ymm0, -80(%rbp)
	vmovupd	%ymm1, -48(%rbp)
L70:
	vmovsd	(%r8,%rcx), %xmm5
	movl	$100, %edx
	xorl	%eax, %eax
	vmovsd	(%rsi,%rcx), %xmm4
	vmovapd	%xmm5, %xmm1
	vmovapd	%xmm4, %xmm0
	.align 4,0x90
L69:
	vmulsd	%xmm0, %xmm0, %xmm0
	vmulsd	%xmm1, %xmm1, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm3
	vucomisd	LC0(%rip), %xmm3
	jbe	L67
	cmpq	%rdx, %rax
	cmovl	%rax, %rdx
L67:
	vaddsd	%xmm4, %xmm0, %xmm0
	addq	$1, %rax
	vsubsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm2
	vfmadd132sd	%xmm2, %xmm5, %xmm1
	cmpq	$100, %rax
	jne	L69
	movq	%rdx, (%rcx,%rdi)
	addq	$8, %rcx
	cmpq	$32, %rcx
	jne	L70
	vzeroupper
	popq	%r10
LCFI13:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI14:
	ret
LFE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE7:
	.text
LHOTE7:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB8:
	.text
LHOTB8:
	.align 4,0x90
	.globl __ZGVdN4vv__Z3foodd
__ZGVdN4vv__Z3foodd:
LFB7:
	leaq	8(%rsp), %r10
LCFI15:
	andq	$-32, %rsp
	xorl	%ecx, %ecx
	pushq	-8(%r10)
	pushq	%rbp
LCFI16:
	movq	%rsp, %rbp
	pushq	%r10
LCFI17:
	leaq	-48(%rbp), %r8
	leaq	-80(%rbp), %rdi
	vmovapd	%ymm0, -80(%rbp)
	vmovapd	%ymm1, -48(%rbp)
	leaq	-112(%rbp), %rsi
L82:
	vmovsd	(%r8,%rcx), %xmm5
	movl	$100, %edx
	xorl	%eax, %eax
	vmovsd	(%rdi,%rcx), %xmm4
	vmovapd	%xmm5, %xmm1
	vmovapd	%xmm4, %xmm0
	.align 4,0x90
L81:
	vmulsd	%xmm0, %xmm0, %xmm0
	vmulsd	%xmm1, %xmm1, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm3
	vucomisd	LC0(%rip), %xmm3
	jbe	L79
	cmpq	%rdx, %rax
	cmovl	%rax, %rdx
L79:
	vaddsd	%xmm4, %xmm0, %xmm0
	addq	$1, %rax
	vsubsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm2
	vfmadd132sd	%xmm2, %xmm5, %xmm1
	cmpq	$100, %rax
	jne	L81
	movq	%rdx, (%rsi,%rcx)
	addq	$8, %rcx
	cmpq	$32, %rcx
	jne	L82
	vmovdqa	-112(%rbp), %ymm0
	popq	%r10
LCFI18:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI19:
	ret
LFE7:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE8:
	.text
LHOTE8:
	.literal8
	.align 3
LC0:
	.long	0
	.long	1074790400
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
	.byte	0x4
	.set L$set$9,LCFI0-LFB3
	.long L$set$9
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$10,LCFI1-LCFI0
	.long L$set$10
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$11,LCFI2-LCFI1
	.long L$set$11
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$12,LCFI3-LCFI2
	.long L$set$12
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$13,LCFI4-LCFI3
	.long L$set$13
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$14,LEFDE9-LASFDE9
	.long L$set$14
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB4-.
	.set L$set$15,LFE4-LFB4
	.quad L$set$15
	.byte	0
	.byte	0x4
	.set L$set$16,LCFI5-LFB4
	.long L$set$16
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$17,LCFI6-LCFI5
	.long L$set$17
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$18,LCFI7-LCFI6
	.long L$set$18
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$19,LCFI8-LCFI7
	.long L$set$19
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$20,LCFI9-LCFI8
	.long L$set$20
	.byte	0xc
	.byte	0x7
	.byte	0x8
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
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$23,LEFDE13-LASFDE13
	.long L$set$23
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB6-.
	.set L$set$24,LFE6-LFB6
	.quad L$set$24
	.byte	0
	.byte	0x4
	.set L$set$25,LCFI10-LFB6
	.long L$set$25
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$26,LCFI11-LCFI10
	.long L$set$26
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$27,LCFI12-LCFI11
	.long L$set$27
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$28,LCFI13-LCFI12
	.long L$set$28
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$29,LCFI14-LCFI13
	.long L$set$29
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$30,LEFDE15-LASFDE15
	.long L$set$30
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB7-.
	.set L$set$31,LFE7-LFB7
	.quad L$set$31
	.byte	0
	.byte	0x4
	.set L$set$32,LCFI15-LFB7
	.long L$set$32
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$33,LCFI16-LCFI15
	.long L$set$33
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$34,LCFI17-LCFI16
	.long L$set$34
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$35,LCFI18-LCFI17
	.long L$set$35
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$36,LCFI19-LCFI18
	.long L$set$36
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE15:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

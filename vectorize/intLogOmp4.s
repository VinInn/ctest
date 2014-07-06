	.text
	.align 4,0x90
	.globl __Z6intLogPfi
__Z6intLogPfi:
LFB243:
	testl	%esi, %esi
	jle	L5
	pushq	%rbx
LCFI0:
	xorl	%edx, %edx
	xorl	%ebx, %ebx
	movl	$8388608, %r9d
	.align 4,0x90
L4:
	movl	(%rdi,%rdx,4), %r8d
	addq	$1, %rdx
	movl	%r8d, %eax
	shrl	$23, %r8d
	andl	$8388607, %eax
	movzbl	%r8b, %r8d
	leal	-127(%rbx,%r8), %ebx
	orq	$8388608, %rax
	imulq	%r9, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	movl	%eax, %ecx
	shrl	$31, %ecx
	addl	%ecx, %ebx
	cmpl	%edx, %esi
	shrx	%ecx, %eax, %r9d
	jg	L4
	shrl	$4, %r9d
	vxorpd	%xmm0, %xmm0, %xmm0
	addl	$8, %ebx
	salq	$19, %r9
	addq	$4194304, %r9
	shrq	$23, %r9
	vcvtsi2sd	%r9d, %xmm0, %xmm0
	call	_log2
	vxorpd	%xmm1, %xmm1, %xmm1
	vcvtsi2sd	%ebx, %xmm1, %xmm1
	popq	%rbx
LCFI1:
	vaddsd	LC1(%rip), %xmm0, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	ret
L5:
	vxorps	%xmm0, %xmm0, %xmm0
	ret
LFE243:
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
	.globl _main
_main:
LFB1496:
	movl	$1024, %ecx
	pushq	%rbp
LCFI2:
	movq	%rcx, %rsi
	movq	%rsp, %rbp
LCFI3:
	andq	$-32, %rsp
	shrq	$3, %rsi
	subq	$4096, %rsp
	leaq	0(,%rsi,8), %rdx
	movq	%rsp, %rdi
	testq	%rdx, %rdx
	je	L17
	cmpq	$7, %rcx
	jbe	L17
	vmovaps	LC2(%rip), %ymm0
	xorl	%eax, %eax
L16:
	movq	%rax, %r8
	addq	$1, %rax
	salq	$5, %r8
	cmpq	%rax, %rsi
	vmovaps	%ymm0, (%rdi,%r8)
	ja	L16
	leaq	(%rdi,%rdx,4), %rax
	cmpq	%rdx, %rcx
	je	L21
	vzeroupper
L11:
	leaq	4(%rax), %rdx
	movl	$0x3f000000, (%rax)
	leaq	4096(%rsp), %rcx
	cmpq	%rdx, %rcx
	je	L14
	leaq	8(%rax), %rdx
	movl	$0x3f000000, 4(%rax)
	cmpq	%rdx, %rcx
	je	L14
	leaq	12(%rax), %rdx
	movl	$0x3f000000, 8(%rax)
	cmpq	%rdx, %rcx
	je	L14
	leaq	16(%rax), %rdx
	movl	$0x3f000000, 12(%rax)
	cmpq	%rdx, %rcx
	je	L14
	leaq	20(%rax), %rdx
	movl	$0x3f000000, 16(%rax)
	cmpq	%rdx, %rcx
	je	L14
	leaq	24(%rax), %rdx
	movl	$0x3f000000, 20(%rax)
	cmpq	%rdx, %rcx
	je	L14
	movl	$0x3f000000, 24(%rax)
L14:
	movl	$1024, %esi
	call	__Z6intLogPfi
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	xorl	%eax, %eax
	leave
LCFI4:
	ret
L17:
LCFI5:
	movq	%rdi, %rax
	jmp	L11
L21:
	vzeroupper
	.p2align 4,,3
	jmp	L14
LFE1496:
	.align 4
__GLOBAL__sub_I_intLogOmp4.cc:
LFB1652:
	leaq	__ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
LCFI6:
	call	__ZNSt8ios_base4InitC1Ev
	movq	__ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	addq	$8, %rsp
LCFI7:
	leaq	___dso_handle(%rip), %rdx
	leaq	__ZStL8__ioinit(%rip), %rsi
	jmp	___cxa_atexit
LFE1652:
	.static_data
__ZStL8__ioinit:
	.space	1
	.literal8
	.align 3
LC1:
	.long	0
	.long	-1070137344
	.const
	.align 5
LC2:
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
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
	.quad	LFB243-.
	.set L$set$2,LFE243-LFB243
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB243
	.long L$set$3
	.byte	0xe
	.byte	0x10
	.byte	0x83
	.byte	0x2
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xc3
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$5,LEFDE3-LASFDE3
	.long L$set$5
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1496-.
	.set L$set$6,LFE1496-LFB1496
	.quad L$set$6
	.byte	0
	.byte	0x4
	.set L$set$7,LCFI2-LFB1496
	.long L$set$7
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$8,LCFI3-LCFI2
	.long L$set$8
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$9,LCFI4-LCFI3
	.long L$set$9
	.byte	0xa
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$10,LCFI5-LCFI4
	.long L$set$10
	.byte	0xb
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$11,LEFDE5-LASFDE5
	.long L$set$11
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB1652-.
	.set L$set$12,LFE1652-LFB1652
	.quad L$set$12
	.byte	0
	.byte	0x4
	.set L$set$13,LCFI6-LFB1652
	.long L$set$13
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$14,LCFI7-LCFI6
	.long L$set$14
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE5:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_intLogOmp4.cc
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

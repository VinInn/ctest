	.text
	.align 4,0x90
	.globl __Z3sumPKfi
__Z3sumPKfi:
LFB4080:
	testl	%esi, %esi
	vxorps	%xmm3, %xmm3, %xmm3
	vmovss	%xmm3, -16(%rsp)
	vmovss	%xmm3, -12(%rsp)
	jle	L2
	vmovaps	%xmm3, %xmm1
	vmovaps	%xmm3, %xmm2
	xorl	%eax, %eax
	jmp	L4
	.align 4,0x90
L6:
	vmovaps	%xmm0, %xmm2
L4:
	vmovss	(%rdi,%rax,4), %xmm0
	addq	$1, %rax
	cmpl	%eax, %esi
	vaddss	%xmm1, %xmm0, %xmm1
	vsubss	%xmm0, %xmm2, %xmm0
	jg	L6
	vmovss	%xmm1, -16(%rsp)
	vmovss	%xmm0, -12(%rsp)
L2:
	vaddss	-16(%rsp), %xmm3, %xmm4
	vaddss	-12(%rsp), %xmm3, %xmm3
	vmovd	%xmm4, %eax
	vmovd	%xmm3, %rdx
	movl	%eax, %eax
	salq	$32, %rdx
	orq	%rdx, %rax
	vmovd	%rax, %xmm0
	ret
LFE4080:
	.align 4,0x90
	.globl __Z4sum4PKfi
__Z4sum4PKfi:
LFB4084:
	leaq	-40(%rsp), %rdx
	leaq	-8(%rsp), %rcx
	movq	%rdx, %rax
L8:
	movl	$0x00000000, (%rax)
	addq	$8, %rax
	movl	$0x00000000, -4(%rax)
	cmpq	%rcx, %rax
	jne	L8
	testl	%esi, %esi
	jle	L10
	leal	-1(%rsi), %eax
	movl	$4, %r9d
	leal	-4(%rsi), %r8d
	andl	$-4, %eax
	subl	%eax, %r8d
	.align 4,0x90
L13:
	cmpl	$4, %esi
	movl	%r9d, %ecx
	cmovle	%esi, %ecx
	xorl	%eax, %eax
	cmpl	%eax, %ecx
	jle	L14
L12:
	vmovss	(%rdi,%rax,4), %xmm0
	vaddss	(%rdx,%rax,8), %xmm0, %xmm1
	vmovss	%xmm1, (%rdx,%rax,8)
	vmovss	4(%rdx,%rax,8), %xmm1
	vsubss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, 4(%rdx,%rax,8)
	addq	$1, %rax
	cmpl	%eax, %ecx
	jg	L12
L14:
	subl	$4, %esi
	addq	$16, %rdi
	cmpl	%r8d, %esi
	jne	L13
L10:
	vmovss	-12(%rsp), %xmm0
	vmovss	-16(%rsp), %xmm2
	vmovss	-36(%rsp), %xmm1
	vmovss	-32(%rsp), %xmm3
	vaddss	-28(%rsp), %xmm1, %xmm1
	vaddss	-20(%rsp), %xmm0, %xmm0
	vaddss	-40(%rsp), %xmm3, %xmm3
	vaddss	-24(%rsp), %xmm2, %xmm2
	vaddss	%xmm0, %xmm1, %xmm0
	vaddss	%xmm2, %xmm3, %xmm2
	vmovss	%xmm0, -12(%rsp)
	vmovss	%xmm2, -16(%rsp)
	vmovq	-16(%rsp), %xmm0
	ret
LFE4084:
	.cstring
LC2:
	.ascii ",\0"
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
	.globl _main
_main:
LFB4514:
	pushq	%rbp
LCFI0:
	vxorps	%xmm0, %xmm0, %xmm0
	pushq	%rbx
LCFI1:
	subq	$4120, %rsp
LCFI2:
	leaq	16(%rsp), %rbx
	vmovss	LC1(%rip), %xmm2
	leaq	4(%rbx), %rax
	movq	%rbx, %rdx
	leaq	4112(%rsp), %rcx
	jmp	L17
	.align 4
L19:
	vmovaps	%xmm1, %xmm0
	addq	$4, %rax
L17:
	cmpq	%rcx, %rax
	vmovss	%xmm0, (%rdx)
	vaddss	%xmm2, %xmm0, %xmm1
	movq	%rax, %rdx
	jne	L19
	movq	%rbx, %rdi
	movl	$1024, %esi
	call	__Z3sumPKfi
	movq	__ZSt4cout@GOTPCREL(%rip), %rbp
	vmovd	%xmm0, %rax
	vmovd	%xmm0, %rsi
	shrq	$32, %rsi
	vmovd	%eax, %xmm3
	vunpcklps	%xmm3, %xmm3, %xmm3
	movl	%esi, 12(%rsp)
	vcvtps2pd	%xmm3, %xmm0
	movq	%rbp, %rdi
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC2(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vmovss	12(%rsp), %xmm0
	movq	%rax, %rdi
	vcvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movl	$1024, %esi
	movq	%rbx, %rdi
	call	__Z4sum4PKfi
	movq	%rbp, %rdi
	vmovd	%xmm0, %rax
	vmovd	%xmm0, %rbx
	shrq	$32, %rbx
	vmovd	%eax, %xmm5
	vunpcklps	%xmm5, %xmm5, %xmm5
	movl	%ebx, 12(%rsp)
	vcvtps2pd	%xmm5, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC2(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vmovss	12(%rsp), %xmm0
	movq	%rax, %rdi
	vcvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	addq	$4120, %rsp
LCFI3:
	xorl	%eax, %eax
	popq	%rbx
LCFI4:
	popq	%rbp
LCFI5:
	ret
LFE4514:
	.align 4
__GLOBAL__sub_I_ured_omp4.cpp:
LFB4777:
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
LFE4777:
	.static_data
__ZStL8__ioinit:
	.space	1
	.literal4
	.align 2
LC1:
	.long	1065353216
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
	.quad	LFB4080-.
	.set L$set$2,LFE4080-LFB4080
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB4084-.
	.set L$set$4,LFE4084-LFB4084
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB4514-.
	.set L$set$6,LFE4514-LFB4514
	.quad L$set$6
	.byte	0
	.byte	0x4
	.set L$set$7,LCFI0-LFB4514
	.long L$set$7
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$8,LCFI1-LCFI0
	.long L$set$8
	.byte	0xe
	.byte	0x18
	.byte	0x83
	.byte	0x3
	.byte	0x4
	.set L$set$9,LCFI2-LCFI1
	.long L$set$9
	.byte	0xe
	.byte	0xb0,0x20
	.byte	0x4
	.set L$set$10,LCFI3-LCFI2
	.long L$set$10
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$11,LCFI4-LCFI3
	.long L$set$11
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$12,LCFI5-LCFI4
	.long L$set$12
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$13,LEFDE7-LASFDE7
	.long L$set$13
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB4777-.
	.set L$set$14,LFE4777-LFB4777
	.quad L$set$14
	.byte	0
	.byte	0x4
	.set L$set$15,LCFI6-LFB4777
	.long L$set$15
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$16,LCFI7-LCFI6
	.long L$set$16
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE7:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_ured_omp4.cpp
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

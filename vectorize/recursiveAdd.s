	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.align 1
	.align 4
	.globl __ZNKSt5ctypeIcE8do_widenEc
	.weak_definition __ZNKSt5ctypeIcE8do_widenEc
__ZNKSt5ctypeIcE8do_widenEc:
LFB3976:
	movl	%esi, %eax
	ret
LFE3976:
	.text
	.align 4,0x90
	.globl __Z3addv
__Z3addv:
LFB3776:
	movss	_c(%rip), %xmm5
	xorl	%eax, %eax
	movss	4+_c(%rip), %xmm4
	leaq	_z(%rip), %r8
	movss	8+_c(%rip), %xmm3
	leaq	_x3(%rip), %rdi
	shufps	$0, %xmm5, %xmm5
	movss	12+_c(%rip), %xmm2
	leaq	_x4(%rip), %rsi
	shufps	$0, %xmm4, %xmm4
	leaq	_x2(%rip), %rcx
	shufps	$0, %xmm3, %xmm3
	leaq	_x1(%rip), %rdx
	shufps	$0, %xmm2, %xmm2
	.align 4,0x90
L3:
	movaps	(%rdi,%rax), %xmm0
	movaps	(%rsi,%rax), %xmm1
	mulps	%xmm3, %xmm0
	mulps	%xmm2, %xmm1
	addps	%xmm1, %xmm0
	movaps	(%rcx,%rax), %xmm1
	mulps	%xmm4, %xmm1
	addps	%xmm1, %xmm0
	movaps	(%rdx,%rax), %xmm1
	mulps	%xmm5, %xmm1
	addps	%xmm1, %xmm0
	movaps	%xmm0, (%r8,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L3
	ret
LFE3776:
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
	.globl _main
_main:
LFB4206:
	movdqa	LC1(%rip), %xmm6
	xorl	%eax, %eax
	pxor	%xmm5, %xmm5
	movdqa	LC0(%rip), %xmm0
	leaq	_x1(%rip), %rdi
	leaq	_x2(%rip), %rsi
	movapd	LC2(%rip), %xmm3
	leaq	_x3(%rip), %rcx
	leaq	_x4(%rip), %rdx
	jmp	L7
	.align 4
L8:
	movdqa	%xmm4, %xmm0
L7:
	movdqa	%xmm0, %xmm2
	cvtdq2ps	%xmm0, %xmm1
	movaps	%xmm1, (%rdi,%rax)
	movdqa	%xmm0, %xmm4
	pslld	$1, %xmm2
	paddd	%xmm6, %xmm4
	movdqa	%xmm2, %xmm1
	pslld	$2, %xmm1
	paddd	%xmm2, %xmm1
	pshufd	$238, %xmm0, %xmm2
	cvtdq2pd	%xmm2, %xmm2
	cvtdq2ps	%xmm1, %xmm1
	mulpd	%xmm3, %xmm2
	movaps	%xmm1, (%rsi,%rax)
	cvtdq2pd	%xmm0, %xmm1
	mulpd	%xmm3, %xmm1
	pslld	$3, %xmm0
	cvtpd2ps	%xmm2, %xmm2
	cvtpd2ps	%xmm1, %xmm1
	movlhps	%xmm2, %xmm1
	movaps	%xmm1, (%rcx,%rax)
	movdqa	%xmm0, %xmm1
	pslld	$2, %xmm1
	paddd	%xmm0, %xmm1
	movdqa	%xmm5, %xmm0
	psubd	%xmm1, %xmm0
	cvtdq2ps	%xmm0, %xmm0
	movaps	%xmm0, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4096, %rax
	jne	L8
	pushq	%r13
LCFI0:
	leaq	4096+_z(%rip), %r13
	pushq	%r12
LCFI1:
	pushq	%rbp
LCFI2:
	pushq	%rbx
LCFI3:
	leaq	_z(%rip), %rbx
	subq	$8, %rsp
LCFI4:
	movaps	LC3(%rip), %xmm0
	movaps	%xmm0, _c(%rip)
	call	__Z3addv
	jmp	L9
	.align 4
L10:
	addq	$4, %rbx
	cmpq	%r13, %rbx
	je	L19
L9:
	movss	(%rbx), %xmm0
	movaps	%xmm0, %xmm1
	andps	LC4(%rip), %xmm1
	cvtss2sd	%xmm1, %xmm1
	comisd	LC5(%rip), %xmm1
	jbe	L10
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	cvtss2sd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %r12
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%r12,%rax), %rbp
	testq	%rbp, %rbp
	je	L20
	cmpb	$0, 56(%rbp)
	je	L13
	movsbl	67(%rbp), %esi
L14:
	movq	%r12, %rdi
	addq	$4, %rbx
	call	__ZNSo3putEc
	movq	%rax, %rdi
	call	__ZNSo5flushEv
	cmpq	%r13, %rbx
	jne	L9
L19:
	addq	$8, %rsp
LCFI5:
	xorl	%eax, %eax
	popq	%rbx
LCFI6:
	popq	%rbp
LCFI7:
	popq	%r12
LCFI8:
	popq	%r13
LCFI9:
	ret
	.align 4
L13:
LCFI10:
	movq	%rbp, %rdi
	call	__ZNKSt5ctypeIcE13_M_widen_initEv
	movq	0(%rbp), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	__ZNKSt5ctypeIcE8do_widenEc@GOTPCREL(%rip), %rax
	je	L14
	movq	%rbp, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	L14
L20:
	call	__ZSt16__throw_bad_castv
LFE4206:
	.align 4
__GLOBAL__sub_I_recursiveAdd.cc:
LFB4468:
	leaq	__ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
LCFI11:
	call	__ZNSt8ios_base4InitC1Ev
	movq	__ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	addq	$8, %rsp
LCFI12:
	leaq	___dso_handle(%rip), %rdx
	leaq	__ZStL8__ioinit(%rip), %rsi
	jmp	___cxa_atexit
LFE4468:
	.static_data
__ZStL8__ioinit:
	.space	1
	.globl _c
	.zerofill __DATA,__pu_bss4,_c,16,4
	.globl _z
	.zerofill __DATA,__pu_bss5,_z,4096,5
	.globl _x4
	.zerofill __DATA,__pu_bss5,_x4,4096,5
	.globl _x3
	.zerofill __DATA,__pu_bss5,_x3,4096,5
	.globl _x2
	.zerofill __DATA,__pu_bss5,_x2,4096,5
	.globl _x1
	.zerofill __DATA,__pu_bss5,_x1,4096,5
	.literal16
	.align 4
LC0:
	.long	0
	.long	1
	.long	2
	.long	3
	.align 4
LC1:
	.long	4
	.long	4
	.long	4
	.long	4
	.align 4
LC2:
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.align 4
LC3:
	.long	1065353216
	.long	1073741824
	.long	3221225472
	.long	1056964608
	.align 4
LC4:
	.long	2147483647
	.long	0
	.long	0
	.long	0
	.literal8
	.align 3
LC5:
	.long	2296604913
	.long	1055193269
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
	.quad	LFB3976-.
	.set L$set$2,LFE3976-LFB3976
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB3776-.
	.set L$set$4,LFE3776-LFB3776
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB4206-.
	.set L$set$6,LFE4206-LFB4206
	.quad L$set$6
	.byte	0
	.byte	0x4
	.set L$set$7,LCFI0-LFB4206
	.long L$set$7
	.byte	0xe
	.byte	0x10
	.byte	0x8d
	.byte	0x2
	.byte	0x4
	.set L$set$8,LCFI1-LCFI0
	.long L$set$8
	.byte	0xe
	.byte	0x18
	.byte	0x8c
	.byte	0x3
	.byte	0x4
	.set L$set$9,LCFI2-LCFI1
	.long L$set$9
	.byte	0xe
	.byte	0x20
	.byte	0x86
	.byte	0x4
	.byte	0x4
	.set L$set$10,LCFI3-LCFI2
	.long L$set$10
	.byte	0xe
	.byte	0x28
	.byte	0x83
	.byte	0x5
	.byte	0x4
	.set L$set$11,LCFI4-LCFI3
	.long L$set$11
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$12,LCFI5-LCFI4
	.long L$set$12
	.byte	0xa
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$13,LCFI6-LCFI5
	.long L$set$13
	.byte	0xc3
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$14,LCFI7-LCFI6
	.long L$set$14
	.byte	0xc6
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$15,LCFI8-LCFI7
	.long L$set$15
	.byte	0xcc
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$16,LCFI9-LCFI8
	.long L$set$16
	.byte	0xcd
	.byte	0xe
	.byte	0x8
	.byte	0x4
	.set L$set$17,LCFI10-LCFI9
	.long L$set$17
	.byte	0xb
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$18,LEFDE7-LASFDE7
	.long L$set$18
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB4468-.
	.set L$set$19,LFE4468-LFB4468
	.quad L$set$19
	.byte	0
	.byte	0x4
	.set L$set$20,LCFI11-LFB4468
	.long L$set$20
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$21,LCFI12-LCFI11
	.long L$set$21
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE7:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_recursiveAdd.cc
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB0:
	.text
LHOTB0:
	.align 4,0x90
	.globl __Z4loadU8__vectorf
__Z4loadU8__vectorf:
LFB0:
	ret
LFE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE0:
	.text
LHOTE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB1:
	.text
LHOTB1:
	.align 4,0x90
	.globl __Z4loadPKf
__Z4loadPKf:
LFB1:
	pxor	%xmm0, %xmm0
	movlps	(%rdi), %xmm0
	movhps	8(%rdi), %xmm0
	ret
LFE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE1:
	.text
LHOTE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB2:
	.text
LHOTB2:
	.align 4,0x90
	.globl __Z5storePfU8__vectorf
__Z5storePfU8__vectorf:
LFB2:
	movlps	%xmm0, (%rdi)
	movhps	%xmm0, 8(%rdi)
	ret
LFE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE2:
	.text
LHOTE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB3:
	.text
LHOTB3:
	.align 4,0x90
	.globl __Z3addPfU8__vectorf
__Z3addPfU8__vectorf:
LFB3:
	pxor	%xmm1, %xmm1
	movlps	(%rdi), %xmm1
	movhps	8(%rdi), %xmm1
	addps	%xmm0, %xmm1
	movlps	%xmm1, (%rdi)
	movhps	%xmm1, 8(%rdi)
	ret
LFE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE3:
	.text
LHOTE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB4:
	.text
LHOTB4:
	.align 4,0x90
	.globl __Z5add11PfS_U8__vectorf
__Z5add11PfS_U8__vectorf:
LFB4:
	movaps	(%rdi), %xmm1
	addps	%xmm0, %xmm1
	movaps	%xmm1, (%rdi)
	addps	(%rsi), %xmm0
	addps	%xmm0, %xmm1
	movaps	%xmm1, (%rsi)
	ret
LFE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE4:
	.text
LHOTE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB5:
	.text
LHOTB5:
	.align 4,0x90
	.globl __Z5add14PfS_U8__vectorf
__Z5add14PfS_U8__vectorf:
LFB5:
	pxor	%xmm1, %xmm1
	movlps	(%rdi), %xmm1
	pxor	%xmm2, %xmm2
	movhps	8(%rdi), %xmm1
	addps	%xmm0, %xmm1
	movlps	%xmm1, (%rdi)
	movhps	%xmm1, 8(%rdi)
	movlps	(%rsi), %xmm2
	movhps	8(%rsi), %xmm2
	addps	%xmm0, %xmm2
	addps	%xmm2, %xmm1
	movlps	%xmm1, (%rsi)
	movhps	%xmm1, 8(%rsi)
	ret
LFE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE5:
	.text
LHOTE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB6:
	.text
LHOTB6:
	.align 4,0x90
	.globl __Z5add98PfS_U8__vectorf
__Z5add98PfS_U8__vectorf:
LFB6:
	pxor	%xmm1, %xmm1
	movlps	(%rdi), %xmm1
	pxor	%xmm2, %xmm2
	movhps	8(%rdi), %xmm1
	addps	%xmm0, %xmm1
	movlps	%xmm1, (%rdi)
	movhps	%xmm1, 8(%rdi)
	movlps	(%rsi), %xmm2
	movhps	8(%rsi), %xmm2
	addps	%xmm0, %xmm2
	addps	%xmm2, %xmm1
	movlps	%xmm1, (%rsi)
	movhps	%xmm1, 8(%rsi)
	ret
LFE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE6:
	.text
LHOTE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB9:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTB9:
	.align 4
	.globl _main
_main:
LFB7:
	pushq	%rbx
LCFI0:
	movl	$12300, %edx
	xorl	%esi, %esi
	subq	$12304, %rsp
LCFI1:
	movq	%rsp, %rdi
	call	_memset
	movaps	LC8(%rip), %xmm1
	movq	%rsp, %rdi
	movl	$0x3f800000, (%rsp)
	leaq	4104(%rsp), %rdx
	movq	%rsp, %rax
	.align 4
L9:
	movlps	%xmm1, (%rax)
	addq	$12, %rax
	movhps	%xmm1, -4(%rax)
	cmpq	%rdx, %rax
	jne	L9
	pxor	%xmm2, %xmm2
	.align 4
L11:
	movaps	%xmm2, %xmm0
	addq	$12, %rdi
	movlps	-12(%rdi), %xmm0
	movhps	-4(%rdi), %xmm0
	addps	%xmm1, %xmm0
	movlps	%xmm0, -12(%rdi)
	movhps	%xmm0, -4(%rdi)
	cmpq	%rdx, %rdi
	jne	L11
	cvttss2si	496(%rsp), %eax
	addq	$12304, %rsp
LCFI2:
	popq	%rbx
LCFI3:
	ret
LFE7:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE9:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTE9:
	.literal16
	.align 4
LC8:
	.long	1065353216
	.long	1073741824
	.long	1077936128
	.long	1082130432
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
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$13,LEFDE13-LASFDE13
	.long L$set$13
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB6-.
	.set L$set$14,LFE6-LFB6
	.quad L$set$14
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$15,LEFDE15-LASFDE15
	.long L$set$15
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB7-.
	.set L$set$16,LFE7-LFB7
	.quad L$set$16
	.byte	0
	.byte	0x4
	.set L$set$17,LCFI0-LFB7
	.long L$set$17
	.byte	0xe
	.byte	0x10
	.byte	0x83
	.byte	0x2
	.byte	0x4
	.set L$set$18,LCFI1-LCFI0
	.long L$set$18
	.byte	0xe
	.byte	0xa0,0x60
	.byte	0x4
	.set L$set$19,LCFI2-LCFI1
	.long L$set$19
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$20,LCFI3-LCFI2
	.long L$set$20
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE15:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

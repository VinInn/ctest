	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB0:
	.text
LHOTB0:
	.align 4,0x90
	.globl __Z4maskU8__vectorf
__Z4maskU8__vectorf:
LFB2464:
	movmskps	%xmm0, %eax
	ret
LFE2464:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE0:
	.text
LHOTE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB1:
	.text
LHOTB1:
	.align 4,0x90
	.globl __Z4maskU8__vectord
__Z4maskU8__vectord:
LFB2465:
	movapd	8(%rsp), %xmm0
	movmskpd	%xmm0, %edx
	movapd	24(%rsp), %xmm0
	movmskpd	%xmm0, %eax
	sall	$2, %eax
	orl	%edx, %eax
	ret
LFE2465:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE1:
	.text
LHOTE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB11:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTB11:
	.align 4
	.globl _main
_main:
LFB3791:
	pushq	%r12
LCFI0:
	pushq	%rbp
LCFI1:
	pushq	%rbx
LCFI2:
	subq	$16, %rsp
LCFI3:
	movq	__ZSt4cout@GOTPCREL(%rip), %r12
	movsd	LC2(%rip), %xmm0
	movq	%r12, %rdi
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	10(%rsp), %rsi
	movl	$1, %edx
	movb	$32, 10(%rsp)
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	pxor	%xmm0, %xmm0
	movq	%rax, %rdi
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movaps	LC4(%rip), %xmm0
	movq	%r12, %rdi
	movmskps	%xmm0, %ebx
	movaps	LC5(%rip), %xmm0
	movmskps	%xmm0, %ebp
	movss	LC6(%rip), %xmm0
	movmskps	%xmm0, %esi
	call	__ZNSolsEi
	leaq	11(%rsp), %rsi
	movl	$1, %edx
	movb	$32, 11(%rsp)
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	%ebp, %esi
	movq	%rax, %rdi
	call	__ZNSolsEi
	movl	$1, %edx
	movb	$32, 12(%rsp)
	leaq	12(%rsp), %rsi
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	%ebx, %esi
	movq	%rax, %rdi
	call	__ZNSolsEi
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%r12, %rdi
	movsd	LC7(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	13(%rsp), %rsi
	movl	$1, %edx
	movb	$32, 13(%rsp)
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	pxor	%xmm0, %xmm0
	movq	%rax, %rdi
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	pxor	%xmm0, %xmm0
	movmskpd	%xmm0, %eax
	movq	%r12, %rdi
	movapd	LC8(%rip), %xmm0
	movmskpd	%xmm0, %ebx
	movsd	LC9(%rip), %xmm0
	sall	$2, %ebx
	movmskpd	%xmm0, %ebp
	orl	%eax, %ebx
	movsd	LC10(%rip), %xmm0
	sall	$2, %ebp
	movmskpd	%xmm0, %esi
	orl	%eax, %ebp
	sall	$2, %eax
	orl	%eax, %esi
	call	__ZNSolsEi
	leaq	14(%rsp), %rsi
	movl	$1, %edx
	movb	$32, 14(%rsp)
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	%ebp, %esi
	movq	%rax, %rdi
	call	__ZNSolsEi
	leaq	15(%rsp), %rsi
	movl	$1, %edx
	movb	$32, 15(%rsp)
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	%ebx, %esi
	movq	%rax, %rdi
	call	__ZNSolsEi
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	addq	$16, %rsp
LCFI4:
	xorl	%eax, %eax
	popq	%rbx
LCFI5:
	popq	%rbp
LCFI6:
	popq	%r12
LCFI7:
	ret
LFE3791:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE11:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTE11:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB12:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTB12:
	.align 4
__GLOBAL__sub_I_movemask.cpp:
LFB4070:
	leaq	__ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
LCFI8:
	call	__ZNSt8ios_base4InitC1Ev
	movq	__ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	leaq	___dso_handle(%rip), %rdx
	addq	$8, %rsp
LCFI9:
	leaq	__ZStL8__ioinit(%rip), %rsi
	jmp	___cxa_atexit
LFE4070:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE12:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTE12:
	.static_data
__ZStL8__ioinit:
	.space	1
	.literal8
	.align 3
LC2:
	.long	3758096384
	.long	-1
	.literal16
	.align 4
LC4:
	.long	0
	.long	0
	.long	0
	.long	4294967295
	.align 4
LC5:
	.long	0
	.long	0
	.long	4294967295
	.long	0
	.align 4
LC6:
	.long	3212836864
	.long	0
	.long	0
	.long	0
	.literal8
	.align 3
LC7:
	.long	4294967295
	.long	-1
	.literal16
	.align 4
LC8:
	.long	0
	.long	0
	.long	4294967295
	.long	-1
	.align 4
LC9:
	.long	4294967295
	.long	-1
	.long	0
	.long	0
	.align 4
LC10:
	.long	0
	.long	-1074790400
	.long	0
	.long	0
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
	.quad	LFB2464-.
	.set L$set$2,LFE2464-LFB2464
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB2465-.
	.set L$set$4,LFE2465-LFB2465
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB3791-.
	.set L$set$6,LFE3791-LFB3791
	.quad L$set$6
	.byte	0
	.byte	0x4
	.set L$set$7,LCFI0-LFB3791
	.long L$set$7
	.byte	0xe
	.byte	0x10
	.byte	0x8c
	.byte	0x2
	.byte	0x4
	.set L$set$8,LCFI1-LCFI0
	.long L$set$8
	.byte	0xe
	.byte	0x18
	.byte	0x86
	.byte	0x3
	.byte	0x4
	.set L$set$9,LCFI2-LCFI1
	.long L$set$9
	.byte	0xe
	.byte	0x20
	.byte	0x83
	.byte	0x4
	.byte	0x4
	.set L$set$10,LCFI3-LCFI2
	.long L$set$10
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$11,LCFI4-LCFI3
	.long L$set$11
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$12,LCFI5-LCFI4
	.long L$set$12
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$13,LCFI6-LCFI5
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
LSFDE7:
	.set L$set$15,LEFDE7-LASFDE7
	.long L$set$15
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB4070-.
	.set L$set$16,LFE4070-LFB4070
	.quad L$set$16
	.byte	0
	.byte	0x4
	.set L$set$17,LCFI8-LFB4070
	.long L$set$17
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$18,LCFI9-LCFI8
	.long L$set$18
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE7:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_movemask.cpp
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

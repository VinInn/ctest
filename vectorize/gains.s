	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
	.align 1
LCOLDB0:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTB0:
	.align 1
	.align 4
	.globl __ZNSt6vectorItSaItEED1Ev
	.weak_definition __ZNSt6vectorItSaItEED1Ev
__ZNSt6vectorItSaItEED1Ev:
LFB788:
	movq	(%rdi), %rdi
	testq	%rdi, %rdi
	je	L1
	jmp	__ZdlPv
	.align 4
L1:
	ret
LFE788:
	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
LCOLDE0:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB2:
	.text
LHOTB2:
	.align 4,0x90
	.globl __Z10applyGainsv
__Z10applyGainsv:
LFB734:
	movq	_ADCs(%rip), %rcx
	movq	8+_ADCs(%rip), %rdi
	movzwl	_firstStrip(%rip), %edx
	cmpq	%rdi, %rcx
	je	L4
	movq	_first(%rip), %r8
	movss	LC1(%rip), %xmm1
	jmp	L6
	.align 4,0x90
L7:
	movl	%esi, %edx
L6:
	movzwl	(%rcx), %eax
	leal	1(%rdx), %esi
	sarl	$7, %edx
	pxor	%xmm0, %xmm0
	movslq	%edx, %rdx
	addq	$2, %rcx
	cvtsi2ss	%eax, %xmm0
	divss	(%r8,%rdx,4), %xmm0
	addss	%xmm1, %xmm0
	cvttss2si	%xmm0, %eax
	movw	%ax, -2(%rcx)
	cmpq	%rcx, %rdi
	jne	L7
L4:
	ret
LFE734:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE2:
	.text
LHOTE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB3:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTB3:
	.align 4
__GLOBAL__sub_I_gains.cc:
LFB793:
	movq	__ZNSt6vectorItSaItEED1Ev@GOTPCREL(%rip), %rdi
	leaq	___dso_handle(%rip), %rdx
	movq	$0, _ADCs(%rip)
	leaq	_ADCs(%rip), %rsi
	movq	$0, 8+_ADCs(%rip)
	movq	$0, 16+_ADCs(%rip)
	jmp	___cxa_atexit
LFE793:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE3:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTE3:
	.globl _first
	.zerofill __DATA,__pu_bss3,_first,8,3
	.globl _firstStrip
	.data
	.align	1
_firstStrip:
	.space	2
	.globl _ADCs
	.zerofill __DATA,__pu_bss4,_ADCs,24,4
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
	.quad	LFB788-.
	.set L$set$2,LFE788-LFB788
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB734-.
	.set L$set$4,LFE734-LFB734
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB793-.
	.set L$set$6,LFE793-LFB793
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_gains.cc
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

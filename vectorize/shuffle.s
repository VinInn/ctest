	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB0:
	.text
LHOTB0:
	.align 4,0x90
	.globl __Z8shuffle1U8__vectorf
__Z8shuffle1U8__vectorf:
LFB0:
	shufps	$177, %xmm0, %xmm0
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
	.globl __Z8shuffle2RKU8__vectorf
__Z8shuffle2RKU8__vectorf:
LFB1:
	movss	12(%rdi), %xmm1
	insertps	$0x10, 8(%rdi), %xmm1
	movss	4(%rdi), %xmm0
	insertps	$0x10, (%rdi), %xmm0
	movlhps	%xmm1, %xmm0
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
	.globl __Z8shuffle3RKU8__vectorf
__Z8shuffle3RKU8__vectorf:
LFB2:
	movaps	(%rdi), %xmm0
	shufps	$177, %xmm0, %xmm0
	ret
LFE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE2:
	.text
LHOTE2:
	.section __TEXT,__text_cold,regular,pure_instructions
	.align 1
LCOLDB3:
	.text
LHOTB3:
	.align 1,0x90
	.align 4,0x90
	.globl __ZNK3foo8shuffle2Ev
__ZNK3foo8shuffle2Ev:
LFB3:
	movss	12(%rdi), %xmm1
	insertps	$0x10, 8(%rdi), %xmm1
	movss	4(%rdi), %xmm0
	insertps	$0x10, (%rdi), %xmm0
	movlhps	%xmm1, %xmm0
	ret
LFE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE3:
	.text
LHOTE3:
	.section __TEXT,__text_cold,regular,pure_instructions
	.align 1
LCOLDB4:
	.text
LHOTB4:
	.align 1,0x90
	.align 4,0x90
	.globl __ZNK3foo8shuffle3Ev
__ZNK3foo8shuffle3Ev:
LFB4:
	movaps	(%rdi), %xmm0
	shufps	$177, %xmm0, %xmm0
	ret
LFE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE4:
	.text
LHOTE4:
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
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

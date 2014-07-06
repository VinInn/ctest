	.text
	.align 4,0x90
	.globl __Z4one1ff
__Z4one1ff:
LFB86:
	mulss	%xmm1, %xmm0
	divss	%xmm1, %xmm0
	cvttss2si	%xmm0, %eax
	ret
LFE86:
	.align 4,0x90
	.globl __Z4one2ff
__Z4one2ff:
LFB87:
	cvttss2si	%xmm0, %eax
	ret
LFE87:
	.align 4,0x90
	.globl __Z5sign1ff
__Z5sign1ff:
LFB88:
	mulss	%xmm1, %xmm0
	movss	LC0(%rip), %xmm2
	andps	%xmm2, %xmm1
	divss	%xmm1, %xmm0
	cvttss2si	%xmm0, %eax
	ret
LFE88:
	.align 4,0x90
	.globl __Z5sign2ff
__Z5sign2ff:
LFB89:
	movss	LC0(%rip), %xmm2
	andps	%xmm1, %xmm2
	divss	%xmm2, %xmm1
	mulss	%xmm0, %xmm1
	cvttss2si	%xmm1, %eax
	ret
LFE89:
	.literal16
	.align 4
LC0:
	.long	2147483647
	.long	0
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
	.quad	LFB86-.
	.set L$set$2,LFE86-LFB86
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB87-.
	.set L$set$4,LFE87-LFB87
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB88-.
	.set L$set$6,LFE88-LFB88
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB89-.
	.set L$set$8,LFE89-LFB89
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

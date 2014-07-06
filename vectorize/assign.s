	.text
	.align 4,0x90
	.globl __Z3barv
__Z3barv:
LFB0:
	movaps	_va(%rip), %xmm1
	movss	_a(%rip), %xmm2
	movaps	%xmm1, %xmm0
	cmpeqps	%xmm1, %xmm0
	shufps	$0, %xmm2, %xmm2
	blendvps	%xmm0, %xmm2, %xmm1
	movaps	%xmm1, _va(%rip)
	ret
LFE0:
	.align 4,0x90
	.globl __Z3foov
__Z3foov:
LFB1:
	movss	_sa(%rip), %xmm0
	movss	_a(%rip), %xmm1
	movaps	%xmm0, %xmm2
	cmpordss	%xmm0, %xmm2
	andps	%xmm2, %xmm1
	andnps	%xmm0, %xmm2
	movaps	%xmm2, %xmm0
	orps	%xmm1, %xmm0
	movss	%xmm0, _sa(%rip)
	ret
LFE1:
	.globl _sa
	.zerofill __DATA,__pu_bss4,_sa,4,4
	.globl _va
	.zerofill __DATA,__pu_bss4,_va,16,4
	.globl _a
	.data
	.align 4
_a:
	.long	1078523331
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
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols

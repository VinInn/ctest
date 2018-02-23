	.file	"fma.cc"
	.text
	.p2align 4,,10
	.p2align 3
	.globl	_Z5goGPUPfS_S_S_
	.type	_Z5goGPUPfS_S_S_, @function
_Z5goGPUPfS_S_S_:
.LFB241:
	.cfi_startproc
	vmovss	36(%rsi), %xmm0
	vmovss	36(%rdx), %xmm3
	vfnmadd132ss	36(%rdi), %xmm3, %xmm0
	vmovss	%xmm0, 36(%rcx)
	vmovss	(%rsi), %xmm0
	vmovss	(%rdx), %xmm4
	vfnmadd132ss	(%rdi), %xmm4, %xmm0
	vmovss	%xmm0, (%rcx)
	vmovss	4(%rsi), %xmm0
	vmulss	4(%rdi), %xmm0, %xmm0
	vaddss	4(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, 4(%rcx)
	vmovss	8(%rsi), %xmm0
	vmulss	8(%rdi), %xmm0, %xmm0
	vsubss	8(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, 8(%rcx)
	vmovss	12(%rdi), %xmm0
	vsubss	12(%rdx), %xmm0, %xmm0
	vmulss	12(%rsi), %xmm0, %xmm0
	vmovss	%xmm0, 12(%rcx)
	vmovss	16(%rdx), %xmm0
	vaddss	16(%rdi), %xmm0, %xmm0
	vmulss	16(%rsi), %xmm0, %xmm0
	vmovss	%xmm0, 16(%rcx)
	vmovss	20(%rdi), %xmm1
	vmulss	.LC0(%rip), %xmm1, %xmm0
	vaddss	.LC1(%rip), %xmm0, %xmm0
	vmulss	%xmm1, %xmm0, %xmm0
	vsubss	.LC2(%rip), %xmm0, %xmm0
	vmulss	%xmm1, %xmm0, %xmm0
	vaddss	.LC3(%rip), %xmm0, %xmm0
	vmulss	%xmm1, %xmm0, %xmm0
	vsubss	.LC4(%rip), %xmm0, %xmm0
	vmulss	%xmm1, %xmm0, %xmm0
	vaddss	.LC5(%rip), %xmm0, %xmm0
	vmulss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, 20(%rcx)
	vmovss	24(%rdi), %xmm1
	vmulss	.LC6(%rip), %xmm1, %xmm0
	vaddss	.LC7(%rip), %xmm0, %xmm0
	vroundss	$1, %xmm0, %xmm0, %xmm0
	vmulss	.LC8(%rip), %xmm0, %xmm2
	vmulss	.LC9(%rip), %xmm0, %xmm0
	vsubss	%xmm2, %xmm1, %xmm1
	vsubss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, 24(%rcx)
	ret
	.cfi_endproc
.LFE241:
	.size	_Z5goGPUPfS_S_S_, .-_Z5goGPUPfS_S_S_
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC0:
	.long	3184084074
	.align 4
.LC1:
	.long	1045305598
	.align 4
.LC2:
	.long	1048820196
	.align 4
.LC3:
	.long	1051395564
	.align 4
.LC4:
	.long	1056958846
	.align 4
.LC5:
	.long	1065352980
	.align 4
.LC6:
	.long	1069066811
	.align 4
.LC7:
	.long	1056964608
	.align 4
.LC8:
	.long	1060205056
	.align 4
.LC9:
	.long	901758606
	.ident	"GCC: (GNU) 6.3.0"
	.section	.note.GNU-stack,"",@progbits

	section .data
str: db `%d\n`, 0

	section .text
	align 4
	global main
	extern printf

; int multiply(int a, int b)
multiply:
	push rbp
	mov rbp, rsp

	mov rax, [rsp+16]
	imul rax, [rsp+24]

	mov rsp, rbp
	pop rbp
	ret

main:
	push rbp
	mov rbp, rsp

	push QWORD 3
	push QWORD 4
	call multiply

	mov rdi, str
	mov rsi, rax
	call printf

	mov rsp, rbp
	pop rbp
	ret

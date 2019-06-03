	section .data
str: db `%d\n`, 0

	section .text
	align 4
	global main
	extern printf

; int multiply(int a, int b)
multiply:
	mov rax, [rsp+32]
	imul rax, [rsp+40]
	mov [rsp+24], rax
	ret

main:
	push rbp
	mov rbp, rsp

	sub rsp, 32
	mov QWORD [rsp+16], 3
	mov QWORD [rsp+8], 4
	push rbp
	push rsp
	call multiply
	pop rsp
	pop rbp

	mov rdi, str
	mov rsi, [rsp]
	call printf

	mov rsp, rbp
	pop rbp
	ret

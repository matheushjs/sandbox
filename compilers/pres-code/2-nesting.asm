	section .data
str: db `%d\n`, 0

	section .text
	align 4
	global main
	extern printf

func1:
	push rbp
	mov rbp, rsp

	mov rax, [rbp+16]
	mov rbx, [rax-8]

	mov rdi, str
	mov rsi, rbx
	call printf

	mov rsp, rbp
	pop rbp
	ret

func2:
	push rbp
	mov rbp, rsp

	mov rax, [rbp+16]
	sub rsp, 16
	mov [rsp], rax
	call func1

	mov rsp, rbp
	pop rbp
	ret

main:
	push rbp
	mov rbp, rsp

	sub rsp, 8
	mov QWORD [rsp], 10

	sub rsp, 8
	mov [rsp], rbp
	call func2

	mov rsp, rbp
	pop rbp
	ret

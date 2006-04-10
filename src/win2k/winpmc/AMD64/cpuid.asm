PUBLIC GetCPUID
.CODE
GetCPUID PROC NEAR
      mov   r8, rdx           ; Save rdx in a register that isn't used by the function.
      mov   r9, rbx           ; Save rbx (it's to be preserved by the called function if modified, and CPUID instruction sets it).
      mov   eax, ecx          ; Set up for reading the CPUID.
      cpuid                   ; Do it..
      mov   dword ptr [r8], eax
      mov   dword ptr [r8+4], ebx
      mov   dword ptr [r8+8], ecx
      mov   dword ptr [r8+12], edx
      mov   rbx, r9
      ret
GetCPUID ENDP
END